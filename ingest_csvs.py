import pandas as pd
import os
from sqlalchemy import create_engine, text
from sqlalchemy.schema import CreateSchema
from settings import usern, passw, db_name, host_machine, port


# pandas options to allow pd dataframes to be easier to read
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_info_columns', 200)
pd.set_option('display.width', 2000)  # Set to the desired display width

# config parameters
schema = 'reservoirs'
res_data_dir = 'res_data'

# creating sqlalchemy engine
engine = create_engine(f'postgresql://{usern}:{passw}@{host_machine}:{port}/{db_name}')


def ingest_csvs_to_postgres(directory):
    '''
    Ingests CSV files from the specified directory into a PostgreSQL database, organizing the data into two tables:
    one for metadata and one for daily storage, precipitation, and temperature values across multiple reservoirs.

    Parameters:
    directory (str): The path to the directory containing the CSV files.


    '''

    # # create schema if it does not exist
    # create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema};"
    # with engine.connect() as connection:
    #     connection.execute(text(create_schema_sql))
    #     print(f'created schema {schema}')


    daily_data_list = []  # List to store DataFrames for daily data
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Loaded {file}")
                df = pd.read_csv(file_path)
                print(df)
                if '_meta.' in file.lower():
                    table_name = file.split('.csv')[0].lower()
                    df.columns = df.columns.str.replace(' ', '_', regex=True)
                    df.to_sql(table_name, engine, if_exists='replace', index=False)
                    print(f"saved meta data to {table_name}\n")
                else:
                    reservoir_name = file.split('_')[1].split('.csv')[0].lower()
                    # creating reservoir_name field
                    df['reservoir_name'] = reservoir_name
                    for col in df.columns:
                        # creating cs_id field and standardizing df for concatenation
                        if 'cs' in col and col.split()[0].isdigit():
                            cs_id = col.split()[0]  # Extract the cs id number
                            df.rename(columns={col: 'cs'}, inplace=True)  # Rename the column to 'cs'
                            df['cs_id'] = cs_id  # Add a new column for cs_id
                            break
                    # appending to daily_data_list for concatenation later
                    df.columns = df.columns.str.replace(' ', '_', regex=True)
                    df['date'] = pd.to_datetime(df['date'])
                    daily_data_list.append(df)

    # concatenating all daily reservoir data into a single DataFrame and then saving to postgresql db
    if daily_data_list:
        print('\nconcatenating all daily reservoir data...')
        all_daily_data = pd.concat(daily_data_list, ignore_index=True)
        print(all_daily_data)
        # Store the concatenated DataFrame in the PostgreSQL table
        all_daily_data.to_sql('reservoir_daily_data', engine, if_exists='replace', index=False)
        print(f"saved daily reservoir data to reservoir_daily_data\n")


def main():
    # ingest data in specific directory
    ingest_csvs_to_postgres(res_data_dir)
    # dispose sqlalchemy engine
    engine.dispose()


if __name__=='__main__':
    main()
