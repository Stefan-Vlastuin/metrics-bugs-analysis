import csv
import os

metrics_path = "../data/metric_values.csv"
bugs_path = "../data/bugs.csv"


def read_csv(path):
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        return [row for row in reader]


# Reads CSV files with metrics values and bugs, and merges them.
# Returns list of dictionaries.
def read_data():
    metrics = read_csv(metrics_path)
    bugs = read_csv(bugs_path)

    data = []
    for row in metrics:
        row: dict
        row['hasBug'] = 1 if has_bug(row['FileName'], bugs) else 0
        data.append(row)

    return data


def has_bug(file_name, bugs):
    for bug_row in bugs:
        bug_row["FileName"] = os.path.basename(bug_row["FileName"])  # Ignore rest of path
        if bug_row["FileName"] == file_name and int(bug_row["Count"]) > 0:
            return True
    return False


# Prepares data for logistic regression.
def prepare_data(data):
    result = dict()
    for key in data[0].keys():
        result[key] = []

    for row in data:
        for (key, value) in row.items():
            result[key].append(value)

    return result