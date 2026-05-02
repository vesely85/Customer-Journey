"""LinkedIn engagement agent for existing facility services connections.

Drafts personalized LinkedIn engagement (post comments, value DMs, light
check-ins) and thought-leadership posts — focused on deepening relationships
and establishing the founder as a credible voice on AI for blue-collar
field-service operations.

Usage:
    python linkedin_agent.py --contact c001 --type value_dm
    python linkedin_agent.py --contact c002 --type comment --context "Their post text..."
    python linkedin_agent.py --contact c003 --type post
    python linkedin_agent.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
CONTACTS_JSON = ROOT / "contacts.json"
CONTACTS_CSV = ROOT / "contacts.csv"

MODEL = "claude-opus-4-7"

ENGAGEMENT_BRIEF = """\
You are a senior advisor helping a founder engage existing 1st-degree LinkedIn
connections in the facility services industry — janitorial, HVAC, landscaping,
security, MEP, property services, and other blue-collar field-service businesses
managing distributed teams.

These are people the founder already knows. There is no introduction to make
and no permission to ask for. The job has two goals, in this order:

  GOAL 1 — DEEPEN THE RELATIONSHIP. Every message gives something away for free
  (an insight, a framework, a benchmark, a useful question). Nothing asks for a
  meeting, a referral, or a sale. Strong relationships compound; treat each
  message as a small, generous deposit.

  GOAL 2 — ESTABLISH THOUGHT LEADERSHIP. The founder is building a reputation
  as the practical voice on AI for blue-collar operations — not a vendor, not
  a hype-merchant. Every interaction should leave the contact slightly smarter
  about their own business. Authority is earned by being specific, being useful,
  and being right about details others gloss over.

The founder's point of view (the spine of every message):

1. AI for blue-collar operations is not chatbots or copilots. It is the removal
   of operational friction that quietly costs facility services companies 5-15
   points of margin every year — and the operators who solve it first will eat
   the operators who don't.

2. The four highest-leverage applications:
   - Workforce scheduling and dispatch that reacts in real time to call-outs,
     traffic, and SLA risk — not a static weekly board.
   - Field QA from photo and video — scoring jobsite work against the spec
     sheet without a regional manager driving 4 hours.
   - Compliance automation: OSHA logs, safety meeting attestations, license
     and cert expirations, DOT/EPA filings — the paperwork that today lives
     in three spreadsheets and a filing cabinet.
   - Predictive maintenance and callback prevention — pattern-matching
     historical work orders to flag the ~8% of jobs that will become callbacks
     before the tech leaves the site.

3. The wedge is always *one* painful workflow, not a platform. Specific beats
   sweeping every time.

4. Tone: peer, not vendor. Concrete, never generic. No emoji. No
   "revolutionize." No "leverage" as a verb. No "in today's fast-paced world."
   No "I hope this finds you well." No exclamation marks. No humble-brags.
   No CTAs. No "let me know if this resonates." No "happy to chat."

DRAFT RULES BY TYPE:

- comment: 1-3 sentences max, posted publicly under their post. Add a sharp
  data point, a specific counter-example, or a precise question that extends
  their thesis. Never "great post." Never restate what they said. The goal
  is to be the comment their network screenshots.

- value_dm: 4-7 short sentences, sent privately. Reference something specific
  about their business or recent activity. Share ONE concrete, useful piece
  of substance — a benchmark, a framework, a tactical observation, a
  question worth sitting with. No ask. End with a period, not a CTA.

- check_in: 2-4 sentences. Warm, no agenda. Reference something specific
  (a post, a hire, a milestone, a shared connection's news). One sentence
  of substance the founder genuinely thinks they'd want to know. End cleanly.

- post: A thought-leadership post for the founder to publish on their OWN
  feed. Audience: operators in facility services who follow them. Use this
  contact's situation as inspiration but write for the broader audience —
  do not name the contact unless explicitly told to. Structure:
    * Hook (one line, scroll-stopping, specific number or contrarian claim)
    * Body (5-12 short lines or a tight list — concrete, no fluff)
    * Closing line (a sharp observation, not a question, not a CTA)
  No hashtags. No "thoughts?" No "what am I missing?" No begging for engagement.

Write as if the founder will copy-paste in 30 seconds. If you cannot be
specific, do not write filler — say what you'd need to know.
"""


def load_contacts() -> dict:
    """Load contacts from contacts.csv if present, otherwise contacts.json.

    CSV format: one row per contact, header row required, columns:
      id, name, title, company, company_type, headcount, relationship,
      recent_activity, known_pain_points, notes

    `known_pain_points` is a single cell with items separated by `;`.
    """
    if CONTACTS_CSV.exists():
        return _load_csv(CONTACTS_CSV)
    if CONTACTS_JSON.exists():
        return json.loads(CONTACTS_JSON.read_text())
    raise SystemExit(f"No contacts file found. Create {CONTACTS_CSV.name} or {CONTACTS_JSON.name}.")


def _load_csv(path: Path) -> dict:
    required = {
        "id", "name", "title", "company", "company_type", "headcount",
        "relationship", "recent_activity", "known_pain_points", "notes",
    }
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path.name} is missing required columns: {sorted(missing)}")
        contacts = []
        for row in reader:
            if not row.get("id"):
                continue  # skip blank rows
            row["known_pain_points"] = [
                p.strip() for p in row["known_pain_points"].split(";") if p.strip()
            ]
            contacts.append(row)
    return {"contacts": contacts}


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
        f"How the founder knows them: {contact['relationship']}",
        f"Recent activity: {contact['recent_activity']}",
        f"Known pain points: {', '.join(contact['known_pain_points'])}",
        f"Notes: {contact['notes']}",
        "",
        f"Draft type: {message_type}",
    ]
    if extra:
        parts += ["", "Additional context:", extra]
    parts += [
        "",
        "Output 3 distinct variants, numbered 1/2/3, each ready to paste. "
        "After the variants, add a one-line note for each explaining what it is "
        "optimized for (e.g. 'specific-number proof', 'contrarian frame', "
        "'tactical generosity').",
    ]
    return "\n".join(parts)


def draft(contact_id: str, message_type: str, extra: str | None) -> str:
    contact = find_contact(contact_id)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
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
    parser = argparse.ArgumentParser(
        description="Draft LinkedIn engagement for existing facility services connections."
    )
    parser.add_argument("--contact", help="Contact id (see --list).")
    parser.add_argument(
        "--type",
        choices=["comment", "value_dm", "check_in", "post"],
        help="Type of draft to generate.",
    )
    parser.add_argument(
        "--context",
        dest="extra",
        help="Optional. For 'comment': the contact's post text. "
        "For 'check_in': the trigger (their hire, milestone, etc.). "
        "For 'post': a specific theme or recent observation to anchor on.",
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
