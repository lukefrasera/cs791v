#!/usr/bin/env python
from HTMLParser import HTMLParser
import sys, os
import subprocess

class CMUFindMocap(HTMLParser):
  def __init__(self):
    HTMLParser.__init__(self)
    self.inLink = False
    self.link = ''
    self.DownloadList = []

  def handle_starttag(self, tag, attr):
    if tag == 'a' and attr[0][0] == 'href':
      self.link = attr[0][1]
  def handle_endtag(self, tag):
    self.inLink = False
  def handle_data(self, data):
    if data == 'amc' or data == 'asf':
      self.DownloadList.append(self.link)




def main():
  with open(sys.argv[1], 'r') as cmu_web_file:
    parser = CMUFindMocap()

    parser.feed(cmu_web_file.read())

    current_dir = os.getcwd()
    os.chdir(sys.argv[2])
    for link in parser.DownloadList:
      subprocess.call("wget {0}".format(link), shell=True)
    os.chdir(current_dir)

# Main Function Caller
if __name__ == '__main__':
  main()