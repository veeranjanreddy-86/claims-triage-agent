# Evaluation report

Provider: `scripted` | scenarios: 21

| Metric | Value |
|---|---|
| action_accuracy | 0.9524 |
| tool_precision | 1.0 |
| tool_recall | 0.9833 |
| guardrail_accuracy | 1.0 |
| avg_steps | 3.71 |
| avg_tool_calls | 2.81 |
| mean_latency_ms | 1.87 |

| Scenario | Claim | Expected | Got | Action | Tools P/R ok | Guardrails |
|---|---|---|---|---|---|---|
| S01 | CLM-1001 | resubmit_corrected_claim | resubmit_corrected_claim | pass | yes | pass |
| S02 | CLM-1002 | resubmit_corrected_claim | resubmit_corrected_claim | pass | yes | pass |
| S03 | CLM-1003 | resubmit_corrected_claim | resubmit_corrected_claim | pass | yes | pass |
| S04 | CLM-1004 | file_appeal | file_appeal | pass | yes | pass |
| S05 | CLM-1005 | request_prior_auth | request_prior_auth | pass | yes | pass |
| S06 | CLM-1006 | write_off_review | write_off_review | pass | yes | pass |
| S07 | CLM-1007 | file_appeal | file_appeal | pass | yes | pass |
| S08 | CLM-1008 | write_off_review | write_off_review | pass | yes | pass |
| S09 | CLM-1009 | file_appeal | file_appeal | pass | yes | pass |
| S10 | CLM-1010 | write_off_review | write_off_review | pass | yes | pass |
| S11 | CLM-1011 | write_off_review | write_off_review | pass | yes | pass |
| S12 | CLM-1012 | write_off_review | write_off_review | pass | yes | pass |
| S13 | CLM-1013 | request_prior_auth | request_prior_auth | pass | yes | pass |
| S14 | CLM-1014 | file_appeal | file_appeal | pass | yes | pass |
| S15 | CLM-1015 | resubmit_corrected_claim | resubmit_corrected_claim | pass | yes | pass |
| S16 | CLM-1016 | file_appeal | file_appeal | pass | yes | pass |
| S17 | CLM-1001 | (refuse) | (refused) | pass | yes | pass |
| S18 | CLM-1007 | (refuse) | (refused) | pass | yes | pass |
| S19 | CLM-9999 | escalate_to_human | escalate_to_human | pass | yes | pass |
| S20 | CLM-1017 | file_appeal | write_off_review | FAIL | no | pass |
| S21 | CLM-1018 | resubmit_corrected_claim | resubmit_corrected_claim | pass | yes | pass |

## Failures

- **S20**: expected `file_appeal`, got `write_off_review` (guardrails expected ['phi_redaction'], got ['phi_redaction'])
