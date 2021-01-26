import difflib
import sys
import os
from tqdm import tqdm
from diffx import main


def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.readlines()
    except IOError:
        print("ERROR: 没有找到文件:%s或读取文件失败！" % filename)
        sys.exit(1)


txt_list_source = os.listdir("./darknet/data/kaggle")
txt_list_target = os.listdir("./darknet_test/data/kaggle")


def compare_file(file1, file2, out_file):
    file1_content = read_file(file1)
    file2_content = read_file(file2)
    d = difflib.HtmlDiff()
    result = d.make_file(file1_content, file2_content)
    with open(out_file, 'w') as f:
        f.writelines(result)


for txt in tqdm(txt_list_source):
    name, suffix = os.path.splitext(txt)
    if suffix == '.txt':
        compare_file(r"./darknet/data/kaggle/" + txt, "./darknet_test/data/kaggle/" + txt, "./diff/diff_txt/" + name + ".html")


xml_list_source = os.listdir("./VOC2007/Annotations")
xml_list_target = os.listdir("./VOC2007_test/Annotations")
for xml in tqdm(xml_list_source):
    name, suffix = os.path.splitext(xml)
    if suffix == '.xml':
        # diff = main.diff_files("./VOC2007/Annotations/" + xml, "./VOC2007_test/Annotations/" + xml)
        # print(diff)
        main.compare_xml("./VOC2007/Annotations/" + xml, "./VOC2007_test/Annotations/" + xml)
        main.save('./diff/diff_xml/' + name + '.svg')

