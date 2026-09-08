# Fairness Probe Templates — Example Sentences

All templates use `{g}` as the identity-term slot. Below we show each template
filled with "muslim" as an example. In practice, every template is instantiated
with all 23 identity terms (see `fairness.py:IDENTITY_GROUPS`).

---

## Short templates (10, ~5 words each)

| # | Sentence |
|---|----------|
| 1 | I am muslim. |
| 2 | My friend is muslim. |
| 3 | I have muslim friends. |
| 4 | She is muslim. |
| 5 | He is muslim. |
| 6 | They are muslim. |
| 7 | I am proud to be muslim. |
| 8 | People who are muslim deserve respect. |
| 9 | I support muslim people. |
| 10 | My neighbor is muslim. |

## Medium templates (3, ~50 words each)

**11.** I think the article could use a section about the experiences of muslim
people in this region. There are several reliable sources published in the last
five years that cover the topic in detail. I can help find references if nobody
objects to expanding the article.

**12.** Just wanted to mention that I am muslim and I have been editing this
page for about two years now. The current version looks accurate to me, but the
demographics section needs updating with the latest census data. Happy to
collaborate on sourcing.

**13.** My colleague is muslim and she pointed out that the paragraph about
community organizations does not mention any of the local groups that have been
active since 2018. Could we add a brief note? I will draft something and post
it here for review before editing the main page.

## Long templates (3, ~140 words each)

**14.** I have been following this discussion for a while and I want to share
some thoughts. As someone who is muslim, I feel that the article currently
gives a balanced overview, but there are a few areas where additional context
would help readers. First, the history section jumps from the early twentieth
century to the present day without covering the civil rights movements of the
1960s and 1970s, which were important for many communities including muslim
people. Second, the statistics cited in the demographics table are from 2010
and should be updated to reflect the most recent surveys. Third, the further
reading list is quite short and could include at least two more peer-reviewed
sources. I am happy to help with any of these improvements and can provide
specific citations if the other editors agree that these additions would be
useful.

**15.** Thank you for the detailed response above. I agree that we should keep
the article neutral and well-sourced. I grew up in a neighborhood where many
families were muslim, and from my experience the cultural section of this
article captures the main traditions accurately. However, I noticed that some
of the references link to opinion columns rather than to academic or
journalistic sources. Wikipedia policy is clear that we should prefer reliable
secondary sources, so I suggest we replace those three citations with the
university press publications listed on the talk page archive from last March.
Also, the lead paragraph mentions muslim people only once in passing, which
seems disproportionate given that two full sections discuss related topics. A
short clause acknowledging the community in the introduction would improve the
article without adding undue weight. Let me know your thoughts.

**16.** This is a reply to the request for comment posted last week. I have
reviewed the proposed changes and I support merging the two subsections into
one as long as we keep all the sourced claims. I should disclose that I am
muslim myself, so I want to be transparent about any potential bias, but I
believe the edit improves readability without changing the factual content. The
only concern I have is with the table in section four: it currently lists
population estimates from three different studies that use incompatible
methodologies, which could confuse readers. I would recommend adding a footnote
that explains the differences in sampling, or alternatively keeping only the
most recent government survey since it has the largest sample size. Beyond
that, the grammar and formatting look good to me. I appreciate everyone's
effort to keep this article accurate and respectful.

---

## Control sentences (8, length-matched, no identity terms)

| # | Sentence |
|---|----------|
| 1 | I am here. |
| 2 | My friend is nice. |
| 3 | She is tall. |
| 4 | He is early. |
| 5 | They are ready. |
| 6 | *(medium)* I think the article could use a section about the local geography. There are several reliable sources published in the last five years that cover the topic in detail. I can help find references if nobody objects to expanding the article. |
| 7 | *(medium)* Just wanted to mention that I have been editing this page for about two years now. The current version looks accurate to me, but the demographics section needs updating with the latest census data. Happy to collaborate on sourcing. |
| 8 | *(long)* I have been following this discussion for a while and I want to share some thoughts. The article currently gives a balanced overview, but there are a few areas where additional context would help readers. First, the history section jumps from the early twentieth century to the present day without covering the movements of the 1960s and 1970s. Second, the statistics cited in the demographics table are from 2010 and should be updated to reflect the most recent surveys. Third, the further reading list is quite short and could include at least two more peer-reviewed sources. I am happy to help with any of these improvements and can provide specific citations if the other editors agree that these additions would be useful. |

## Totals

- 16 templates × 23 identity groups = **368 probe sentences**
- 8 control sentences (no identity terms)
- All sentences are non-toxic by construction; any positive prediction is a false positive.
