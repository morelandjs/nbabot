NBA model
=========

*Framework to calibrate an Elo regressor on NBA boxscore data*

Quick start
-----------

Install the project requirements with pip::

   pip3 install -r requirements.txt

Populate the sqlite database with NBA schedule and boxscore data ::

  python3 -m src.data

Train the ``elora`` regressor on the specified statistic, e.g. first-half line ::

  python3 -m src.model

Validate the model predictions to ensure their statistical veracity ::

  python3 -m src.validate

Generate model predictions using the calibrated model ::

  python3 -m src.predict rank
  python3 -m src.predict forecast

See ``python3 -m src.predict --help`` for options and details!
