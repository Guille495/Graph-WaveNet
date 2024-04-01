import pandas as pd
import numpy as np


def generate_training_data(dfs=dfs, seq_length_x=10,seq_length_y=10,y_start=1,add_month=True,add_season=True,add_day=True,pp_list=True,predictors_list=[]):

  x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
  y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

  # Divide the datasets in x and y and transform the input
  for cat in ["train", "val", "test"]:
    x, y, dates, stations = generate_graph_seq2seq_io_data_V3(
          df=dfs[cat],
          x_offsets=x_offsets,
          y_offsets=y_offsets,
          scaler=None,
          #df_pp=df_pp,
          add_month=add_month,
          add_season=add_season,
          add_day=add_day,
          pp_list=pp_list,
          predictors_list=predictors_list
      )

    print(cat, "x: ", x.shape, "y:", y.shape, "date", len(dates))

    np.savez_compressed(
        os.path.join("./output", f"{cat}.npz"),
        x=x,
        y=y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        dates=dates,
        stations=stations
    )


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, scaler=None, add_month=False, add_season=False, add_day=False , pp_list = True , predictors_list = [] , predictors_df = selected_predictors_df
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param scaler:
    :param add_month:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    dates = [pd.to_datetime(i).strftime('%Y-%m-%d') for i in df.index]
    stations = df.columns
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)

    feature_list = [data]

    if (add_month):
      months = [(pd.to_datetime(i).month - 1)/ 11 for i in df.index]
      month_of_day = np.tile(months, [1, num_nodes, 1]).transpose((2, 1, 0))
      feature_list.append(month_of_day)

    if(add_season):
      season = [(pd.to_datetime(i).month % 4)/ 3 for i in df.index]
      season = np.tile(season, [1, num_nodes, 1]).transpose((2, 1, 0))
      feature_list.append(season)


    if(add_day):
      days_of_year = [(pd.to_datetime(i).day_of_year - 1)/355 for i in df.index]
      days_of_year = np.tile(days_of_year, [1, num_nodes, 1]).transpose((2, 1, 0))
      feature_list.append(days_of_year)

    if(pp_list):

      if df_pp.shape[1] != num_nodes:
        data_pp = df_pp.pivot(columns='station_id', values='pp')

      else:
        data_pp = df_pp.copy()

      data_pp = data_pp.fillna(0)
      data_pp = data_pp.loc[df.index]
      data_pp = np.expand_dims(data_pp.values, axis=-1)
      data_pp = (data_pp - data_pp.min())/(data_pp.max() - data_pp.min())

      #data = np.concatenate([data, data_pp], axis=2)
      feature_list.append(data_pp)


    if predictors_list:

      for FEATURE in predictors_list:

        df_feature = predictors_df.pivot(columns='station_id', values=FEATURE)
        df_feature = df_feature.fillna(0)
        df_feature = df_feature.loc[df.index]
        df_feature = np.expand_dims(df_feature.values, axis=-1)
        df_feature = (df_feature - df_feature.min()) / (df_feature.max() - df_feature.min())

        feature_list.append(df_feature)


    data = np.concatenate(feature_list, axis=-1)
    x, y, dates = [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive


    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        dates.append(df.index[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    return x, y, dates, stations


def generate_training_data_V3(dfs=dfs, seq_length_x=10,seq_length_y=10,y_start=1,add_month=True,add_season=True,add_day=True,pp_list=True,predictors_list=[]):

  x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
  y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

  # Divide the datasets in x and y and transform the input
  for cat in ["train", "val", "test"]:
    x, y, dates, stations = generate_graph_seq2seq_io_data_V3(
          df=dfs[cat],
          x_offsets=x_offsets,
          y_offsets=y_offsets,
          scaler=None,
          #df_pp=df_pp,
          add_month=add_month,
          add_season=add_season,
          add_day=add_day,
          pp_list=pp_list,
          predictors_list=predictors_list
      )

    print(cat, "x: ", x.shape, "y:", y.shape, "date", len(dates))

    np.savez_compressed(
        os.path.join("./output", f"{cat}.npz"),
        x=x,
        y=y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        dates=dates,
        stations=stations
    )
