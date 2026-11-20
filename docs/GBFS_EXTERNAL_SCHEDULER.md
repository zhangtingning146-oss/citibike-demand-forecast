# GBFS external scheduler fallback

## Status checked on 2026-11-20

GitHub's native `schedule` event eventually began firing. Six genuine scheduled runs succeeded between 2026-11-18 21:38 UTC and 2026-11-20 08:41 UTC, and Release assets now cover three UTC dates. However, the automatic gaps were roughly 5.2 to 8.4 hours rather than 15 minutes. The current conclusion is therefore: **automatic execution and Release persistence verified; 15-minute service level not verified**.

The downloaded Release assets contain nine distinct feed times, 22,669 station-status rows, about 2,519 stations per snapshot, and zero duplicate `(feed_last_updated_utc, station_id)` keys. The external scheduler is still recommended to supply the missing cadence; it should coexist with the native schedule because the collector deduplicates safely.

GitHub's native `schedule` event remained inactive even though the repository is public, the workflow is active on the default branch, Actions are enabled, and manual dispatch succeeds. The collector and GitHub Release persistence are retained unchanged.

The lowest-permission practical fallback is a free external HTTPS scheduler such as cron-job.org calling GitHub's workflow-dispatch API every 15 minutes:

```text
POST https://api.github.com/repos/zhangtingning146-oss/citibike-demand-forecast/actions/workflows/schedule-gbfs.yml/dispatches
Authorization: Bearer <fine-grained token stored only in the scheduler secret store>
Accept: application/vnd.github+json
Content-Type: application/json

{"ref":"master","inputs":{"trigger_source":"cron-job.org"}}
```

The fine-grained token should be restricted to this one public repository with only **Actions: read and write** permission and a short expiration. It must never be committed, placed in a URL query string, saved in project files, or pasted into a report/chat. Configure the external service's secret header field directly in its authenticated web interface.

The workflow run title includes `GBFS / cron-job.org`, so an external dispatch can be distinguished from a user manual dispatch even though both use GitHub's `workflow_dispatch` event.

## Minimum user actions

1. In GitHub, create a fine-grained personal access token with a short expiry, select only `zhangtingning146-oss/citibike-demand-forecast`, and grant only **Repository permissions → Actions: Read and write**. Metadata read access is automatic. Do not paste the token into chat or project files.
2. Create a free cron-job.org account and a job scheduled every 15 minutes. Use the POST URL and JSON body above.
3. Add the three request headers shown above in cron-job.org's header/secret fields. Put the token only in the `Authorization: Bearer ...` header. Never put it in the URL, body, job title, or logs.
4. Enable the job. Verification requires two `GBFS / cron-job.org` runs roughly 15 minutes apart, both showing Release growth and zero duplicate keys.

This fallback cannot be activated by project code alone: the repository owner must create the external scheduler account and token, then store the token in that service. Until that is done, the accurate status is “native automatic collection verified but severely sparse; 15-minute external cadence not yet activated.”
