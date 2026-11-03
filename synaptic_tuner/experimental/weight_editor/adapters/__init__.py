"""Experimental, unqualified ``ModelAdapter`` implementations for the weight editor.

Nothing here is a Tuner API v1 promise. Adapters are trusted operator
composition: the caller supplies the ``Executor`` separately and the broker
still owns admission, budgets, write-ahead dispatch and the journal.
"""
