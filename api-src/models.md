# Models

The domain data, all frozen dataclasses. Nothing here knows about a database, a provider or a
web request.

## Documents, pages and chunks

```{eval-rst}
.. automodule:: ragsage.models
   :members:
```

## Scope

The engine's only isolation concept, and the boundary that lets one engine serve a script and
a multi-tenant SaaS unchanged. A pricing, auth or tenancy change must never require editing
this file — that is the litmus test for the boundary.

```{eval-rst}
.. automodule:: ragsage.scope
   :members:
```
