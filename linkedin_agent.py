"""LinkedIn engagement agent for facility services contacts.

Drafts personalized LinkedIn outreach (connection notes, intro DMs, post comments,
follow-ups) anchored on AI for blue-collar workforce efficiency and compliance.

Usage:
    python linkedin_agent.py --contact c001 --type intro_dm
    python linkedin_agent.py --contact c002 --type comment --post "Their post text..."
    python linkedin_agent.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
CONTACTS_PATH = ROOT / "contacts.json"

MODEL = "claude-opus-4-7"

ENGAGEMENT_BRIEF = """\
You are a senior advisor helping a founder engage their LinkedIn network in the
facility services industry — janitorial, HVAC, landscaping, security, MEP,
property services, and other blue-collar field-service businesses managing
distributed teams.

The founder's point of view (use as the spine of every message):

1. AI for blue-collar operations is not chatbots or copilots — it is the
   removal of operational friction that quietly costs facility services
   companies 5-15 points of margin every year.

2. The four highest-leverage applications:
   - Workforce scheduling and dispatch that reacts to call-outs, traffic,
     and SLA risk in real time (not a static weekly board).
   - Field QA from photo and video — scoring jobsite work against the
     spec sheet without a regional manager driving 4 hours.
   - Compliance automation: OSHA logs, safety meeting attestations,
     license/cert expirations, DOT/EPA filings — the paperwork that today
     lives in three spreadsheets and a filing cabinet.
   - Predictive maintenance and callback prevention — pattern-matching
     historical work orders to flag the 8% of jobs that will become
     callbacks before the tech leaves the site.

3. The wedge is *one* painful workflow, not a platform. "We replaced your
   inspection paperwork" beats "we'll transform your operations."

4. Tone: peer, not vendor. Concrete, never generic. No emoji. No
   "revolutionize." No "leverage." No "in today's fast-paced world."
   No "I hope this finds you well." No exclamation marks.

Hard rules for what you draft:
- LinkedIn connection notes: 280 characters MAX, hard cap. Reference one
  specific thing about them. Do not pitch.
- Intro DMs: 4-6 short sentences. One specific observation about their
  business or a recent post. One concrete insight or question tied to AI
  for their workflow. No call-to-action heavier than "would you find that
  useful?" or "open to comparing notes?"
- Post comments: 1-2 sentences that add a sharp data point or question.
  Never "great post!" Never repeat their thesis back at them.
- Follow-ups: assume they didn't reply. Reference what changed since the
  last message — a new datapoint, a peer's result, a relevant news item.
  Never guilt or reference their silence.

Write as if the founder will copy-paste this into LinkedIn in 30 seconds.
"""


def load_contacts() -> dict:
    return json.loads(CONTACTS_PATH.read_text())


def find_contact(contact_id: str) -> dict:
    for c in load_contacts()["contacts"]:
        if c["id"] == contact_id:
            return c
    raise SystemExit(f"Contact {contact_id!r} not found. Run with --list to see options.")


def build_user_prompt(contact: dict, message_type: str, extra: str | None) -> str:
    parts = [
        f"Contact: {contact['name']}, {contact['title']} at {contact['company']}",
        f"Company type: {contact['company_type']}",
        f"Scale: {contact['headcount']}",
        f"Relationship: {contact['relationship']}",
        f"Recent activity: {contact['recent_activity']}",
        f"Known pain points: {', '.join(contact['known_pain_points'])}",
        f"Notes: {contact['notes']}",
        "",
        f"Draft type: {message_type}",
    ]
    if extra:
        parts += ["", f"Additional context for this draft:", extra]
    parts += [
        "",
        "Output 3 distinct variants, numbered 1/2/3, each ready to paste. "
        "After the variants, add a one-line note explaining what each variant "
        "is optimized for (e.g. 'curiosity hook', 'peer-to-peer', "
        "'specific-number proof').",
    ]
    return "\n".join(parts)


def draft(contact_id: str, message_type: str, extra: str | None) -> str:
    contact = find_contact(contact_id)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": ENGAGEMENT_BRIEF,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_prompt(contact, message_type, extra)}],
    )

    out = []
    for block in response.content:
        if block.type == "text":
            out.append(block.text)
    return "\n".join(out).strip()


def list_contacts() -> None:
    for c in load_contacts()["contacts"]:
        print(f"  {c['id']}  {c['name']:<22} {c['title']:<30} {c['company']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft LinkedIn outreach for facility services contacts.")
    parser.add_argument("--contact", help="Contact id (see --list).")
    parser.add_argument(
        "--type",
        choices=["connection_note", "intro_dm", "comment", "follow_up"],
        help="Type of draft to generate.",
    )
    parser.add_argument(
        "--post",
        dest="extra",
        help="For 'comment': the contact's post text. For 'follow_up': what changed since last message.",
    )
    parser.add_argument("--list", action="store_true", help="List available contacts and exit.")
    args = parser.parse_args()

    if args.list:
        list_contacts()
        return

    if not args.contact or not args.type:
        parser.error("--contact and --type are required (or pass --list).")

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in.")

    print(draft(args.contact, args.type, args.extra))


if __name__ == "__main__":
    main()
