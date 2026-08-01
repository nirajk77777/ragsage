# Ports

Thirteen `Protocol`s, in pipeline order. An adapter conforms by *shape*: it imports and
subclasses nothing from ragsage, so the dependency arrow only ever points inward.

```{eval-rst}
.. automodule:: ragsage.ports
   :members:
```

## Small talk

The gate in front of retrieval. A greeting was never a question about the corpus, so
reporting a miss on it would be a lie about what the engine did.

```{eval-rst}
.. automodule:: ragsage.smalltalk
   :members:
```
