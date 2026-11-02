import uuid

import httpx
import pytest

from app.debtors import CSV_COLUMNS, CsvImportError, parse_csv
from app.policy.config import load_rules
from tests.conftest import MakeDebtor

HEADER = ",".join(CSV_COLUMNS)
RULES = load_rules()


def _ref() -> str:
    return f"I-{uuid.uuid4().hex[:10]}"


def test_parse_accepts_required_columns_only() -> None:
    rows = parse_csv(
        "ref,full_name,dob,balance_jpy,mobile\nA-1,Ren Example,1991-06-15,95000,090\n", rules=RULES
    )
    assert rows[0].balance_jpy == 95000
    assert rows[0].timezone == "Asia/Tokyo"
    assert (rows[0].min_monthly_jpy, rows[0].max_months) == (None, None)


def test_customer_plan_limits_parse_when_stricter() -> None:
    rows = parse_csv(
        "ref,full_name,dob,balance_jpy,mobile,min_monthly_jpy,max_months\n"
        "A-1,Ren Example,1991-06-15,95000,090,8000,4\n",
        rules=RULES,
    )
    assert (rows[0].min_monthly_jpy, rows[0].max_months) == (8000, 4)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ("0,4", "row 2: invalid min_monthly_jpy"),
        ("5000,7", "row 2: max_months must be 3-6"),  # looser than the global 6 months
        ("5000,2", "row 2: max_months must be 3-6"),  # shorter than any allowed plan
    ],
)
def test_customer_plan_limits_looser_than_global_are_rejected(limits: str, message: str) -> None:
    with pytest.raises(CsvImportError) as exc:
        parse_csv(
            "ref,full_name,dob,balance_jpy,mobile,min_monthly_jpy,max_months\n"
            f"A-1,Ren Example,1991-06-15,95000,090,{limits}\n",
            rules=RULES,
        )
    assert message in str(exc.value)


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        ("ref,full_name,dob\nA-1,X,1990-01-01\n", "missing columns: balance_jpy"),
        ("ref,full_name,dob,balance_jpy,salary\n", "unknown columns: salary"),
        (
            "ref,full_name,dob,balance_jpy,mobile\nA-1,X,1990-01-01,-5,090\n",
            "row 2: invalid balance_jpy",
        ),
        ("ref,full_name,dob,balance_jpy\nA-1,X,1990-01-01,100\n", "row 2: invalid row"),
        (
            "ref,full_name,dob,balance_jpy,mobile,timezone\nA-1,X,1990-01-01,1,090,Mars/Base\n",
            "row 2: invalid timezone",
        ),
        (
            "ref,full_name,dob,balance_jpy,mobile\nA-1,X,1990-01-01,1,090\nA-1,Y,1990-01-01,1,090\n",
            "row 3: duplicate ref",
        ),
        ("ref,full_name,dob,balance_jpy\n", "no rows"),
    ],
)
def test_parse_rejects_bad_files_without_echoing_values(csv_text: str, message: str) -> None:
    with pytest.raises(CsvImportError) as exc:
        parse_csv(csv_text, rules=RULES)
    assert message in str(exc.value)
    assert "Mars" not in str(exc.value)  # values may be PII: messages name fields only


async def test_import_adds_new_refs_skips_existing_and_shows_the_card(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    existing, new = await make_debtor(), _ref()
    body = (
        f"{HEADER}\n"
        f"{existing},Someone Else,1990-01-01,1,Asia/Tokyo,090,,,false,false,,\n"
        f"{new},Ren Example,1991-06-15,95000,America/New_York,+1-555,,,false,true,,"
        "Says it was paid\n"
    )
    resp = await api_client.post(
        "/debtors/import", content=body, headers={"Content-Type": "text/csv"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"added": [new], "skipped": [existing]}
    assert new in (await api_client.get("/debtors")).json()
    card = (await api_client.get(f"/debtors/{new}")).json()
    assert card["full_name"] == "Ren Example"
    assert card["balance_jpy"] == 95000
    assert card["timezone"] == "America/New_York"
    assert card["has_open_dispute"] is True
    assert card["no_workplace_contact"] is True
    assert card["persona"] is None
    # Unchanged: an existing ref is never overwritten.
    assert (await api_client.get(f"/debtors/{existing}")).json()["full_name"] == "Test Person"


async def test_bad_file_adds_nothing(api_client: httpx.AsyncClient) -> None:
    good, bad = _ref(), _ref()
    body = (
        "ref,full_name,dob,balance_jpy,mobile\n"
        f"{good},A,1990-01-01,1,090\n{bad},B,not-a-date,1,090\n"
    )
    resp = await api_client.post(
        "/debtors/import", content=body, headers={"Content-Type": "text/csv"}
    )
    assert resp.status_code == 422
    assert (await api_client.get(f"/debtors/{good}")).status_code == 404


async def test_debtor_routes_need_the_token(api_client: httpx.AsyncClient) -> None:
    resp = await api_client.get("/debtors", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


async def test_eval_and_live_debtors_are_hidden_from_testers(
    api_client: httpx.AsyncClient, make_debtor: MakeDebtor
) -> None:
    mine, machine = await make_debtor(), await make_debtor(ref=f"EV-{uuid.uuid4().hex[:10]}")
    refs = (await api_client.get("/debtors")).json()
    assert mine in refs
    assert machine not in refs
