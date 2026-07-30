# Configuration

Every knob arrives in a frozen dataclass constructed by *you*, at your own composition root.
The library reads no environment variables, here or anywhere — see
[ADR-0002](../adr/0002-ragsage-owns-its-storage.md). A config object whose values depend on
the ambient process is untestable, un-injectable and silently different in production.

## Pipeline policy

```{eval-rst}
.. automodule:: ragsage.config
   :members:
```

## Providers

```{eval-rst}
.. automodule:: ragsage.providers.config
   :members:
```

## Postgres

```{eval-rst}
.. automodule:: ragsage.storage.config
   :members:
```
