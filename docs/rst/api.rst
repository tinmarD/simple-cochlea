====================
    API Reference
====================

References for `simplecochlea` classes

.. contents::
   :local:
   :depth: 2

:py:mod:`simplecochlea`:

.. automodule:: simplecochlea
   :no-members:
   :no-inherited-members:

Cochlea Sub-Classes
====================

.. currentmodule:: simplecochlea

.. autosummary::

   cochlea.BandPassFilterbank
   cochlea.RectifierBank
   cochlea.CompressionBank
   cochlea.LIFBank

Cochlea Class
=============

:py:mod:`simplecochlea.cochlea.Cochlea`:

.. currentmodule:: simplecochlea.cochlea.Cochlea

.. autosummary::

   process_input
   process_input_block_ver
   process_one_channel
   process_test_signal
   plot_channel_evolution
   plot_filterbank_frequency_response
   save

Test signals
============

:py:mod:`simplecochlea.generate_signals`:

.. currentmodule:: simplecochlea.generate_signals

.. autosummary::

   generate_sinus
   generate_dirac
   generate_step
   merge_wav_sound_from_dir
   generate_abs_stim
   get_abs_stim_params
   delete_zero_signal


SpikeList
=========

:py:mod:`simplecochlea.spikes.spikelist`:

.. currentmodule:: simplecochlea.spikes.spikelist.SpikeList

.. autosummary::

   sort
   select_spike
   epoch
   epoch_on_triggers
   export
   to_dataframe
   plot
   plot_channel_selectivity
   get_median_isi
   get_mean_isi
   set_pattern_id_from_time_limits
   get_pattern_results
   get_channel_selectivity
   add_time_offset



