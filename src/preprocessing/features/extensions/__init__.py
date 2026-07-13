"""Feature-group extensions.

Drop a module here that defines a concrete ``FeatureGroup`` subclass and it
will be auto-discovered by ``FeatureGroupRegistry`` and included by the
``FeatureEngineer`` (when enabled via a preset's ``enable_groups`` or the
``"all"`` sentinel). Declare ``feature_prefixes`` / ``feature_keywords`` on
the class so the ``FeatureSelector`` keeps the group's features.
"""