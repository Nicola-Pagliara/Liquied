from EDA import EDA as eda


def main():
    eda.overview_analysis(original_path='Dataset/time_series.xlsx', name_data=['15min', '30min', '60min'])


if __name__ == '__main__':
    main()
