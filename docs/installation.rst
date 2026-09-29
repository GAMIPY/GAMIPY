.. highlight:: shell

============
Installation
============


Stable release
--------------

To install GAMIPY, run this command in your terminal:

.. code-block:: console

    $ pip install GAMIPY

This is the preferred method to install GAMIPY, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for GAMIPY can be downloaded from the `GitHub repo`_:

.. code-block:: console

    $ git clone https://github.com/GAMIPY/GAMIPY.git
    $ cd GAMIPY
    $ pip install .

For development (editable install plus the test, lint and docs tools),
see :doc:`contributing`.

GAMIPY supports Python 3.10 to 3.13 and depends on pandas, NumPy, SciPy and
scikit-learn, which ``pip`` installs automatically.

.. _GitHub repo: https://github.com/GAMIPY/GAMIPY
