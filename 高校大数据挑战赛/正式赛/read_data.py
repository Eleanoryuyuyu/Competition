import csv


def gen_stream_data(data_path):
    with open('labeled_content', 'w') as f:
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                query_id = row[0]
                query = row[1]
                title = row[3]
                label = row[4]
                line_count+=1
                if line_count >=10000*10000: # 10000*10000 一亿条要一晚上
                    f.write("{0},{1},{2},{3}\n".format(query_id, query, title, label))
                    #break
            print(f'Processed {line_count} lines.')


