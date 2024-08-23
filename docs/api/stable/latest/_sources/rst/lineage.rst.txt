
Lineage
=======

.. automodule:: mlopus.lineage

.. autofunction:: mlopus.lineage.of

.. autoclass:: mlopus.lineage.Lineage
   :members: register, with_input_model, with_input_artifact, with_output_model, with_output_artifact
   :member-order: bysource

   .. autoattribute:: inputs
   .. autoattribute:: outputs

.. autoclass:: mlopus.lineage.Inputs
   :members:

   .. autoattribute:: mlopus.lineage._LineageInfo::models
      :noindex:
   .. autoattribute:: mlopus.lineage._LineageInfo::runs
      :noindex:
   .. autoattribute:: mlopus.lineage._LineageInfo::runs_by_path
      :noindex:

.. autoclass:: mlopus.lineage.Outputs
   :members:

   .. autoattribute:: mlopus.lineage._LineageInfo::models
      :noindex:
   .. autoattribute:: mlopus.lineage._LineageInfo::runs
      :noindex:
   .. autoattribute:: mlopus.lineage._LineageInfo::runs_by_path
      :noindex:

.. autoclass:: mlopus.lineage.Query
   :members: render, with_input_model, with_input_artifact, with_output_model, with_output_artifact
   :member-order: bysource
