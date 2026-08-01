# Façades

Four entry points. The first three are the primitives; `RagSage` assembles them.

## Ingestion

```{eval-rst}
.. automodule:: ragsage.ingestion
   :members:
```

## Query

```{eval-rst}
.. automodule:: ragsage.query
   :members:
```

## Evaluation

```{eval-rst}
.. automodule:: ragsage.evaluation
   :members:
```

```{eval-rst}
.. automodule:: ragsage.goldens
   :members:
```

## The assembler

Not re-exported from the top-level package, deliberately: importing it pulls SQLAlchemy,
asyncpg and the provider SDKs, and `import ragsage` is guaranteed to stay free of that stack.

```{eval-rst}
.. automodule:: ragsage.sage
   :members:
```
