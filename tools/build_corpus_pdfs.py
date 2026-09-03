"""Generate the two PDF corpus documents from source text.

Kept in tools/ (not ingested) so the PDFs in corpus/ are reproducible rather
than opaque binaries checked into the repo.

    python tools/build_corpus_pdfs.py
"""

from __future__ import annotations

import pathlib

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

CORPUS = pathlib.Path(__file__).resolve().parent.parent / "corpus"

# --- styles -----------------------------------------------------------------

_base = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_base["Heading1"], fontSize=16, spaceAfter=10, spaceBefore=4)
H2 = ParagraphStyle("H2", parent=_base["Heading2"], fontSize=12.5, spaceAfter=7, spaceBefore=13)
H3 = ParagraphStyle("H3", parent=_base["Heading3"], fontSize=10.8, spaceAfter=5, spaceBefore=9)
BODY = ParagraphStyle(
    "BODY", parent=_base["BodyText"], fontSize=9.6, leading=13.6, alignment=TA_LEFT, spaceAfter=5
)
MONO = ParagraphStyle("MONO", parent=BODY, fontName="Courier", fontSize=8.2, leading=10.6)


def render(text: str, out_path: pathlib.Path, title: str) -> None:
    """Render a lightly-marked-up plain text document to PDF.

    Markup: '# ' h1, '## ' h2, '### ' h3, '| ' preformatted, '---' page break.
    """
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        title=title,
        author="Meridian Academy",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    flow = []
    for raw in text.strip("\n").split("\n"):
        line = raw.rstrip()
        if line == "---":
            flow.append(PageBreak())
        elif not line:
            flow.append(Spacer(1, 4))
        elif line.startswith("### "):
            flow.append(Paragraph(line[4:], H3))
        elif line.startswith("## "):
            flow.append(Paragraph(line[3:], H2))
        elif line.startswith("# "):
            flow.append(Paragraph(line[2:], H1))
        elif line.startswith("| "):
            flow.append(Paragraph(line[2:].replace(" ", "&nbsp;"), MONO))
        else:
            # Escape stray ampersands for reportlab's mini-XML parser, but put
            # back the one entity the source text uses intentionally. Without
            # the second replace, "&nbsp;" becomes "&amp;nbsp;" and renders as
            # the literal string "&nbsp;" in the PDF — which then lands in the
            # extracted text and pollutes the very header lines that carry the
            # document ID.
            safe = line.replace("&", "&amp;").replace("&amp;nbsp;", "&nbsp;")
            flow.append(Paragraph(safe, BODY))
    doc.build(flow)
    print(f"wrote {out_path.relative_to(CORPUS.parent)}  ({out_path.stat().st_size:,} bytes)")


# --- document 1: course brochure --------------------------------------------

BROCHURE = """
# Meridian Academy — Course Brochure 2026

Document ID: MA-BROCHURE-2026-02 &nbsp;|&nbsp; Version 5.0 &nbsp;|&nbsp; Effective 2026-02-01
Owner: Academic Office &nbsp;|&nbsp; Audience: Prospective learners and admissions counsellors

## About Meridian Academy

Meridian Academy is an online, cohort-based upskilling institution for working professionals
and career starters. We were founded in 2019 and have graduated approximately 14,200 learners
across five programs. All instruction is live and instructor-led; we are not a self-paced video
platform. Attendance, assessment completion, and project submission are tracked, and they are
preconditions for our placement services.

We do not operate physical campuses. All classes are delivered online. Our registered office is
at Level 6, Kestrel Tower, Outer Ring Road, Bengaluru 560103.

Meridian Academy is not a university. We do not award degrees or UGC-recognised diplomas. On
successful completion, learners receive a Meridian Academy Certificate of Completion, which is
an industry certificate.

## How our programs are structured

Every Meridian program follows the same structural spine, regardless of subject:

Live classes are held on weekday evenings from 8:00 PM to 10:00 PM IST and on Saturdays from
10:00 AM to 1:00 PM IST. Every session is recorded and published to the platform within 12 hours.
Recordings remain accessible for 12 months after the program end date.

Each program is divided into six modules. Each module ends with a graded module assessment, passed
at 60%. Learners get two free reattempts per assessment. Three failures route the learner into
Academic Support, a four-week mentored remediation track offered at no additional cost.

Every program includes at least two capstone projects, reviewed and graded by an instructor.
Capstone projects must be submitted and approved to clear the Placement Readiness Gate.

Mentor groups are capped at 12 learners. Cohorts are capped at 120 learners.

Expect a weekly commitment of 15 to 18 hours: roughly 7 hours of live class, 4 hours of
assignments, 3 hours of project work, and 2 hours of mentor and doubt-resolution sessions.
DSML-301 and ASD-501 trend towards the upper end of that range.

---

## SEF-101 — Software Engineering Foundations

Duration: 9 months &nbsp;|&nbsp; Level: Beginner &nbsp;|&nbsp; Cohorts: Monthly
List tuition: INR 2,10,000 exclusive of GST

SEF-101 is our entry program and the only Meridian program that requires no degree and no prior
coding experience. It is designed for career changers, non-engineering graduates, and Class 12
completers who want a first structured route into software engineering.

### Syllabus

### Module 1 — Programming Foundations (6 weeks)
Python syntax and semantics, control flow, functions, error handling, file I/O, modules and
packages, virtual environments, and an introduction to version control with Git.

### Module 2 — Data Structures (7 weeks)
Arrays, strings, linked lists, stacks, queues, hash tables, trees, heaps, and graphs. Emphasis on
implementation from first principles before use of library types.

### Module 3 — Algorithms and Complexity (7 weeks)
Asymptotic analysis, sorting and searching, recursion, two pointers, sliding window, binary search
on answer, greedy methods, dynamic programming, and graph traversal.

### Module 4 — Databases and SQL (5 weeks)
Relational modelling, normalisation, SQL querying and joins, indexing, transactions, and an
introduction to non-relational stores.

### Module 5 — Backend Web Development (7 weeks)
HTTP, REST API design, request handling, authentication and authorisation, ORMs, caching basics,
and deploying a service.

### Module 6 — Engineering Practice (4 weeks)
Testing, debugging, code review, CI basics, reading unfamiliar codebases, and an introduction to
system design at interview level.

### Capstone projects
Two required. Project A is a command-line application with persistent storage. Project B is a
deployed REST API with authentication, tests, and a written design note.

### Career outcomes
Typical target roles are Junior Software Engineer, Backend Engineer (entry), and Software
Development Engineer I. SEF-101 includes full placement support services. SEF-101 is not eligible
for the Placement Assurance conditional refund product.

---

## FSWD-201 — Full-Stack Web Development

Duration: 11 months &nbsp;|&nbsp; Level: Intermediate &nbsp;|&nbsp; Cohorts: Monthly
List tuition: INR 2,85,000 exclusive of GST

FSWD-201 takes learners who already have some coding exposure and builds production-grade
full-stack capability. It requires a bachelor's degree (or final-year enrolment) and at least six
months of demonstrable coding exposure.

### Syllabus

### Module 1 — Modern JavaScript and TypeScript (6 weeks)
ES2023 language features, asynchronous programming, modules, the type system, generics, and
tooling.

### Module 2 — Frontend Engineering with React (8 weeks)
Component architecture, hooks, state management, routing, forms, data fetching, accessibility, and
performance profiling.

### Module 3 — Backend Services with Node (8 weeks)
Express and Fastify, middleware, validation, authentication with JWT and sessions, file handling,
background jobs, and WebSockets.

### Module 4 — Data Layer (6 weeks)
PostgreSQL modelling and query tuning, Redis for caching and queues, migrations, connection
pooling, and an introduction to MongoDB.

### Module 5 — System Design for Web Applications (7 weeks)
Load balancing, horizontal scaling, caching strategies, rate limiting, idempotency, eventual
consistency, and observability.

### Module 6 — Delivery and Operations (6 weeks)
Docker, CI/CD pipelines, deployment on cloud platforms, monitoring, logging, incident response,
and cost awareness.

### Capstone projects
Three required. A production-quality full-stack application, a public API with documented
contracts and rate limiting, and a performance optimisation case study on a supplied slow codebase.

### Career outcomes
Typical target roles are Full-Stack Engineer, Frontend Engineer, and Backend Engineer. FSWD-201
includes full placement support and is one of only two programs eligible for Placement Assurance,
with a minimum assured CTC of INR 7,00,000.

---

## DSML-301 — Data Science and Machine Learning

Duration: 13 months &nbsp;|&nbsp; Level: Advanced &nbsp;|&nbsp; Cohorts: Every two months
List tuition: INR 3,40,000 exclusive of GST

DSML-301 is our longest and most demanding program. It requires a bachelor's degree with a
quantitative component, at least twelve months of full-time work experience, and a technical
screening call. Learners receive USD 400 of cloud GPU credits for model training; no local GPU is
required.

### Syllabus

### Module 1 — Mathematics for Machine Learning (7 weeks)
Linear algebra, calculus for optimisation, probability, and statistical inference including
hypothesis testing and confidence intervals.

### Module 2 — Data Engineering and Analysis (8 weeks)
Python data stack, SQL at analytical scale, data cleaning, feature engineering, exploratory
analysis, and visualisation.

### Module 3 — Classical Machine Learning (9 weeks)
Linear and logistic regression, regularisation, decision trees, random forests, gradient boosting,
support vector machines, clustering, dimensionality reduction, and rigorous model evaluation.

### Module 4 — Deep Learning (9 weeks)
Neural network fundamentals, backpropagation, convolutional networks for vision, recurrent and
attention-based architectures for sequence data, transfer learning, and training at scale.

### Module 5 — Natural Language Processing and Large Language Models (8 weeks)
Text representation, transformer architectures, fine-tuning, prompt engineering, retrieval
augmented generation, evaluation of generative systems, and responsible deployment.

### Module 6 — Machine Learning in Production (7 weeks)
Experiment tracking, model packaging and serving, feature stores, monitoring and drift detection,
A/B testing, and cost and latency management.

### Capstone projects
Three required. An end-to-end supervised learning project with a written evaluation report, a deep
learning project on vision or language, and a deployed inference service with monitoring.

### Career outcomes
Typical target roles are Data Scientist, Machine Learning Engineer, and Applied Scientist
(entry to mid). DSML-301 includes full placement support and is eligible for Placement Assurance,
with a minimum assured CTC of INR 7,00,000.

---

## DCE-401 — DevOps and Cloud Engineering

Duration: 8 months &nbsp;|&nbsp; Level: Intermediate &nbsp;|&nbsp; Cohorts: Every two months
List tuition: INR 2,45,000 exclusive of GST

DCE-401 is for professionals already working in an IT role who want to move into infrastructure,
platform, or reliability engineering. It requires twelve months of IT experience and comfort with
the Linux command line. Learners receive USD 250 of cloud lab credits.

### Syllabus

### Module 1 — Linux and Networking Foundations (5 weeks)
Process and memory management, filesystems, permissions, shell scripting, TCP/IP, DNS, HTTP, TLS,
and troubleshooting methodology.

### Module 2 — Infrastructure as Code (5 weeks)
Terraform, state management, modules, drift detection, and configuration management with Ansible.

### Module 3 — Containers and Orchestration (7 weeks)
Docker internals, image optimisation, Kubernetes architecture, workloads, services, ingress,
storage, autoscaling, and Helm.

### Module 4 — CI/CD and Release Engineering (5 weeks)
Pipeline design, artefact management, testing gates, blue-green and canary deployment, feature
flags, and rollback strategy.

### Module 5 — Observability and Reliability (6 weeks)
Metrics, logs, traces, service level objectives, error budgets, alerting design, on-call practice,
and incident postmortems.

### Module 6 — Cloud Architecture and Security (6 weeks)
Compute and storage services, networking and VPC design, identity and access management, secrets
handling, cost optimisation, and compliance basics.

### Capstone projects
Two required. A fully automated multi-environment infrastructure deployment, and an observability
and incident-response exercise on a deliberately fragile system.

### Career outcomes
Typical target roles are DevOps Engineer, Site Reliability Engineer, Cloud Engineer, and Platform
Engineer. DCE-401 includes full placement support. DCE-401 is not eligible for Placement Assurance.

---

## ASD-501 — Advanced Systems Design

Duration: 6 months &nbsp;|&nbsp; Level: Senior &nbsp;|&nbsp; Cohorts: Quarterly (Jan, Apr, Jul, Oct)
List tuition: INR 1,95,000 exclusive of GST

ASD-501 is a senior specialisation for engineers with at least four years of professional software
engineering experience, of which at least two years must involve building or operating production
backend systems. The Meridian Aptitude Test is waived; admission is decided by a 45-minute
technical interview with a senior instructor. Approximately 40% of applicants are declined at that
stage.

ASD-501 is explicitly not a career-switching program. It includes no placement services of any
kind and is not eligible for Placement Assurance. Applicants who need placement support should
choose a different program.

### Syllabus

### Module 1 — Foundations of Distributed Systems (4 weeks)
Failure models, consistency models, the CAP and PACELC framings, consensus, clocks and ordering,
and quorum systems.

### Module 2 — Data at Scale (5 weeks)
Storage engine internals, replication and partitioning strategies, distributed transactions,
change data capture, and polyglot persistence.

### Module 3 — Event-Driven Architecture (4 weeks)
Log-based messaging, exactly-once semantics in practice, stream processing, event sourcing, CQRS,
and the saga pattern.

### Module 4 — Performance Engineering (4 weeks)
Latency budgets, queueing theory for engineers, profiling, load and stress testing, capacity
planning, and back-of-envelope estimation.

### Module 5 — Resilience and Operability (4 weeks)
Timeouts, retries and backoff, circuit breakers, bulkheads, graceful degradation, chaos
experiments, and designing for on-call.

### Module 6 — Architecture in Practice (5 weeks)
Trade-off analysis, architecture decision records, migration and strangler patterns, cost and
organisational constraints, and leading a design review.

### Capstone projects
Two required. A full design document for a large-scale system with an explicit trade-off analysis,
and a live design review presented to and critiqued by a panel of senior instructors.

### Career outcomes
ASD-501 is intended to support internal progression to Senior Engineer, Staff Engineer, and
Architect roles. Meridian Academy provides no placement services for this program.

---

## Instructors

Meridian instructors are practising engineers, not full-time academics. Every instructor has at
least six years of industry experience and passes an internal teaching evaluation before leading a
cohort. Instructors are assigned per module, so learners are taught each subject by a specialist
rather than by a single generalist across the whole program.

Mentor groups of at most 12 learners meet weekly. Mentors are separate from instructors and are
responsible for progress tracking, project feedback, and escalation of learners who fall behind.

## Certification

On completion of all six module assessments, the required capstone projects, and the final
technical evaluation, learners receive a Meridian Academy Certificate of Completion bearing a
verifiable credential ID. Certificates are issued within 21 days of program completion. A reissue
costs INR 1,500.

The certificate is an industry certificate. It is not a degree and is not accredited by the UGC or
AICTE.

## Next steps

Applications are made at the Meridian Academy website. Admission requires the Meridian Aptitude
Test, and for DSML-301, DCE-401, and ASD-501 a technical screening call. Full requirements are in
the Eligibility Criteria document, MA-ELIG-2026-01. Pricing, discounts, and financing terms are in
the pricing master, MA-PRICING-2026-03. Placement services and the Placement Assurance product are
described in the Placement Policy, MA-PLACE-2026-02.
"""


# --- document 2: placement policy -------------------------------------------

PLACEMENT = """
# Meridian Academy — Placement Policy

Document ID: MA-PLACE-2026-02 &nbsp;|&nbsp; Version 3.4 &nbsp;|&nbsp; Effective 2026-02-01
Owner: Career Services &nbsp;|&nbsp; Status: Authoritative for all placement matters

## 1. Purpose and scope

This document defines what placement support Meridian Academy provides, who qualifies for it, and
the precise terms of the Placement Assurance conditional refund product. It applies to all learners
enrolled in cohorts starting on or after 1 February 2026.

Meridian Academy does not guarantee employment. Nothing in this document should be read as a
promise of a job. We provide structured placement support, and separately we offer a conditional
refund product on two programs, subject to strict participation conditions set out in Section 5.

## 2. Which programs include placement services

| Program    Placement support   Placement Assurance eligible
| ---------  ------------------  ---------------------------
| SEF-101    Yes                 No
| FSWD-201   Yes                 Yes
| DSML-301   Yes                 Yes
| DCE-401    Yes                 No
| ASD-501    No                  No

ASD-501 is a senior specialisation and includes no placement services of any kind.

## 3. The Placement Readiness Gate

Placement services do not begin when a learner enrols, and they do not begin when the program ends.
They begin when the learner clears the Placement Readiness Gate, referred to throughout as the PRG.

### 3.1 PRG requirements

A learner clears the PRG on the date all three of the following are simultaneously true:

(a) Attendance credit of at least 85%. Attending a live session counts as 1.0 credit. Watching the
full recording of a missed session within 7 calendar days counts as 0.5 credit. Sessions neither
attended live nor watched within 7 days count as zero.

(b) All six module assessments passed, at the 60% pass mark, in any number of permitted attempts.

(c) The required number of capstone projects for the program submitted and approved by an
instructor. This is two projects for SEF-101 and DCE-401, and three for FSWD-201 and DSML-301.

### 3.2 What happens if a learner does not clear the PRG

Learners who do not clear the PRG continue to receive career coaching, résumé review, mock
interviews, and access to the learning platform. They are not referred to hiring partners and are
not eligible for Placement Assurance. There is no penalty beyond that, and a learner may clear the
PRG later and enter placement support at that point.

### 3.3 PRG audit

PRG status is computed automatically and audited manually by Career Services before the first
hiring partner referral. Where the audit finds that attendance or project approval was recorded in
error, the PRG date is corrected and the learner is informed in writing within 5 business days.

## 4. Placement support services

Learners who have cleared the PRG receive the following for 12 months from their PRG clearance
date. The window runs from PRG clearance, not from program end date, so a learner who clears the
PRG late still receives a full 12-month window.

Résumé and portfolio review, with up to three rounds of written feedback.

Mock interviews: up to eight sessions, comprising technical, system design, and behavioural rounds,
conducted by practising engineers.

Hiring partner referrals to companies in the Meridian partner network, matched to the learner's
program, experience level, and stated location preference.

Access to the Meridian job board, which lists roles from partner and non-partner companies.

Salary negotiation guidance from a career coach at the offer stage.

Two structured career-strategy sessions, one at PRG clearance and one at the six-month mark.

## 5. Placement Assurance

Placement Assurance is a conditional refund product, not a job guarantee. It is available only on
FSWD-201 and DSML-301.

### 5.1 What it promises

If an eligible learner who has complied with every condition in Section 5.3 does not receive a
qualifying job offer within 12 months of clearing the PRG, Meridian Academy refunds 80% of tuition
actually paid, net of scholarships and discounts, exclusive of GST.

### 5.2 What counts as a qualifying job offer

A qualifying job offer is a written offer of full-time employment, from a company with at least
25 employees, for a role in software engineering, data science, or a directly adjacent technical
discipline, at a location in India or fully remote from India, at an annual cost to company of at
least INR 7,00,000.

An internship, a contract engagement shorter than 12 months, an unpaid role, a commission-only
role, and an offer from a company in which the learner or an immediate family member holds an
ownership interest are all excluded.

### 5.3 Learner conditions

Placement Assurance applies only if the learner satisfies every one of the following. These
conditions are cumulative and are strictly enforced.

(a) Cleared the PRG within 60 calendar days of the program end date.

(b) Applied to at least 30 roles referred by Career Services during the 12-month window.

(c) Attended at least 10 interviews arranged through Career Services. A no-show without at least
24 hours' notice counts against this total as a breach rather than an attendance.

(d) Did not decline two or more qualifying job offers as defined in Section 5.2. Declining one
qualifying offer is permitted; declining a second ends Placement Assurance immediately.

(e) Maintained a total scholarship discount of 30% of tuition or less. Learners with a stacked
scholarship discount above 30% receive full placement support but are not eligible for Placement
Assurance.

(f) Responded to Career Services communications within 5 business days throughout the window.

(g) Is located in India and legally entitled to work in India. Learners located outside India are
not eligible.

(h) Did not accept employment outside the covered disciplines and then request the refund.

### 5.4 How to claim

A claim is submitted to placement-claims@meridianacademy.example within 30 calendar days of the end
of the 12-month window. Claims received after that deadline are not considered.

The claim must include the learner's application log, interview record, and a written statement.
Career Services verifies the claim against its own records within 15 business days. Where records
conflict, the Meridian platform record governs.

Approved refunds are disbursed within 30 business days of approval, following the payment routing
rules in the Refund Terms document, MA-REFUND-2026-01. Where tuition was financed, the refund is
remitted to the partner NBFC first.

### 5.5 Interaction with the standard refund policy

Placement Assurance is separate from, and does not extend, the standard refund schedule in
MA-REFUND-2026-01. A learner who withdraws from the program is not eligible for Placement
Assurance, regardless of how much of the program they completed.

## 6. Hiring partner network

The Meridian hiring partner network comprises approximately 340 companies as of February 2026,
across four segments: venture-funded product startups, mid-market SaaS companies, IT services and
consulting firms, and a small number of large product companies.

Partner status is reviewed annually. A partner that makes no hire and conducts no interviews for
two consecutive years is removed from the network.

Referrals are matched, not broadcast. Career Services refers a learner to roles matching their
program, demonstrated skill level, and stated location preference. A learner may request referral
to a specific partner, and Career Services will make the referral where the learner meets the
stated role requirements.

## 7. Reported outcomes

The following figures cover learners who cleared the PRG in the twelve months ending 31 December
2025 and are published for transparency. They are historical and are not a prediction.

| Program    PRG clearers   Placed in 12 mo   Median CTC (INR)   Highest CTC (INR)
| ---------  -------------  ----------------  -----------------  ----------------
| SEF-101    1,842          71%               6,40,000           18,00,000
| FSWD-201   1,516          78%               9,20,000           31,00,000
| DSML-301     938          74%              11,80,000           42,00,000
| DCE-401      604          76%              10,40,000           28,00,000

"Placed" means the learner reported accepting a written full-time offer within 12 months of PRG
clearance. Figures are self-reported by learners and verified by Career Services against offer
letters where the learner supplies one. Approximately 8% of PRG clearers did not respond to
outcome surveys and are excluded from the percentages, which therefore overstate placement rates to
an unknown but bounded degree.

Median CTC is the median of reported accepted offers, not of all offers received.

## 8. Learner obligations during placement

A learner receiving placement support must keep their profile and résumé current, respond to
referral communications within 5 business days, attend scheduled interviews or give at least 24
hours' notice, report offers received and accepted within 7 calendar days, and represent their
experience and credentials honestly.

Misrepresenting experience, qualifications, or Meridian coursework to a hiring partner is a code of
conduct violation. It results in immediate withdrawal of placement services, forfeiture of
Placement Assurance, and, where the misrepresentation is material, notification to the affected
hiring partner.

## 9. Suspension and reinstatement of placement services

Placement services are suspended where a learner's payment account is in arrears by more than 30
days, where a code of conduct investigation is open, or where the learner has been unresponsive to
Career Services for more than 30 calendar days.

Suspension pauses the 12-month placement window. Reinstatement restarts it with the remaining
balance of the window intact. A learner may be reinstated at most twice.

## 10. Escalation

Placement disputes are escalated to careers-grievance@meridianacademy.example. The Career Services
lead responds within 10 business days. Placement Assurance claim rejections may be appealed once,
within 15 calendar days of the rejection, to the Head of Career Services, whose decision is final.
"""


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    render(BROCHURE, CORPUS / "course_brochure.pdf", "Meridian Academy Course Brochure 2026")
    render(PLACEMENT, CORPUS / "placement_policy.pdf", "Meridian Academy Placement Policy")


if __name__ == "__main__":
    main()
