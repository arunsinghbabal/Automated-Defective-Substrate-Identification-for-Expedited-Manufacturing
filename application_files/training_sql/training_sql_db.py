import sqlite3
import csv
import os


class SqlDb:
    def __init__(self, par_path):
        self.par_path = par_path
        self.sql_path = os.path.join(par_path, 'sql_db/')

    def database(self, sch_col_name, log, log_path):
        """
            Method Name: create_database
            Description: This function creates an SQL database with a table containing the columns from the
             syntax file and rows from all the good raw files and save it later as a csv file in the
             combined_csv folder.
            Output: None
        """
        log_file = open(log_path + 'sql_db.txt', 'a+')
        train_good_path = os.path.join(self.par_path, 'train_raw_good/')
        train_combined_csv_dir = os.path.join(self.par_path, 'train_combined_csv/')
        try:
            # create a SQL database
            con = sqlite3.connect(self.sql_path + 'sql_train' + '.db')
            c = con.cursor()
            # Check if the table with a name train_good_raw_dt exist or not
            c.execute("SELECT COUNT(NAME) FROM sqlite_master WHERE TYPE = 'table' AND NAME = 'train_good_raw_dt'")
            dt_exist = c.fetchone()[0]
            if dt_exist == 0:
                # If train_good_raw_dt table does not exist in the database than we first create the table using the
                # except block and then add the column and it type one by one in try block using the for loop
                for k, v in sch_col_name.items():
                    try:
                        # Add the column name and its type in the table
                        c.execute("ALTER TABLE train_good_raw_dt ADD COLUMN '{col_name}' {col_type}".format(col_name=k,
                                                                                                            col_type=v))
                        log.log(log_file, 'Added col names in the table')
                    except:
                        # Create a table named train_good_raw_dt
                        c.execute(
                            "CREATE TABLE train_good_raw_dt ({col_name} {col_type})".format(col_name=k, col_type=v))
                        log.log(log_file, 'Table created successfully')
                # List all the files in raw_good folder
                train_good_files = [f for f in os.listdir(train_good_path)]
                # Pick the files one by one and insert their content in the train_good_raw_dt table
                for file in train_good_files:
                    with open(os.path.join(train_good_path, file), 'r') as f:
                        next(f)
                        reader = csv.reader(f, delimiter='\n')  # Read the opened csv file
                        # reader is a collection of rows, where rows are in string format
                        for line in reader:
                            # Here line is a list with only one string value (row), line=['cola,colb,....']
                            for l in line:
                                # Here l is just a string so we have to separate it using the spilt and then put it
                                # into a tuple because insert takes only a tuple
                                csv_val = tuple(l.split(','))
                                c.execute("INSERT INTO train_good_raw_dt values {value}".format(value=csv_val))
                                # Commit the changes in the table
                                con.commit()
                # Select all the rows in table train_good_raw_dt
                c.execute("SELECT * FROM train_good_raw_dt")
                # fetch all the table data in the variable
                dt_data = c.fetchall()
                # c.description gives the detail list of col such as (column name,type,....)
                dt_col_name = [i[0] for i in c.description]
                csv_final = csv.writer(open(train_combined_csv_dir + 'train_unprocessed_input.csv', 'w', newline=''),
                                       delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_ALL, escapechar='\\')
                csv_final.writerow(dt_col_name)
                # Write the rows one by one in the input.csv file using the csv_final writer object
                for x in dt_data:
                    csv_final.writerow(x)
            else:
                c.close()
            log_file.close()

        except Exception as e:
            # Call the log function to append the error message
            log.log(log_file, 'Error during SQL database operations')
            log.log(log_file, str(e))
            log_file.close()
