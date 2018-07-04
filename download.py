# Multi purpose downloader script
# 
# - requests library for HTTP operations
# - standard library's queue library 
# - standard library's threads library

import os
import queue
import threading
import sys
import getopt
import requests
import json
import time


BAD_URLS = []

#Downloader class - reads queue and downloads each file in succession
class Downloader(threading.Thread):
    """Threaded File Downloader"""

    def __init__(self, fqueue, output_directory):
            threading.Thread.__init__(self, name= os.urandom(16))
            self.fqueue = fqueue
            self.output_directory = output_directory

    def run(self):
        while True:
            # gets the url from the queue
            url = self.fqueue.get()

            # download the file
            # print("* Thread " + self.name + " - processing URL")
            self.download_file(url)

            # send a signal to the queue that the job is done
            self.fqueue.task_done()

    def download_file(self, url):
        t_start = time.clock()

        r = requests.get(url)
        if (r.status_code == requests.codes.ok):
            t_elapsed = time.clock() - t_start
            # print("* Thread: " + self.name + " Downloaded " + url + " in " + str(t_elapsed) + " seconds")
            fname = self.output_directory + "/" + os.path.basename(url)
            if not os.path.isfile(fname):
                with open(fname, "wb") as f:
                    f.write(r.content)
        else:
            print("* Thread: " + self.name + " Bad URL: " + url)
            BAD_URLS.append(url)


# Spawns dowloader threads and manages URL downloads queue
class DownloadManager():

    def __init__(self, download_dict, output_directory, thread_count=8):
        self.thread_count = thread_count
        self.download_dict = download_dict
        self.output_directory = output_directory

    # Start the downloader threads, fill the queue with the URLs and
    # then feed the threads URLs via the queue
    def begin_downloads(self):
        fqueue = queue.Queue()

        # Create a thread pool and give them a queue
        for i in range(self.thread_count):
            t = Downloader(fqueue, self.output_directory)
            t.setDaemon(True)
            t.start()

        # Load the queue from the download dict
        for linkname in self.download_dict:
            #print uri
            fqueue.put(self.download_dict[linkname])

        # Wait for the queue to finish
        fqueue.join()

        return


# Main.  Parse CLIoptions, prepare download list & start downloading
def main():
    output_directory = "train"
    flist = []
    with open("trainEnglish.txt", "r") as f:
        for line in f:
            flist.append(line.split(',')[1])

    help = 'python download.py'

    print('----------pydownnload---------------')
    print('------------------------------------')
    print('Output Directory:      ', output_directory)
    print('File number:           ', len(flist))
    print('------------------------------------')

    # Now build a dict of urls to download, just add any flist urls
    download_dict = {}

     # Add in any additional files contained in the flist variable
    if (flist is not None):
        for f in flist:
            download_dict[str(f)] = f

    # If there are no URLs to download then exit now, nothing to do!
    if len(download_dict) is 0:
        print("* No URLs to download, got the usage right?")
        print("USAGE: " + help)
        sys.exit(2)

    download_manager = DownloadManager(download_dict, output_directory, 5)
    download_manager.begin_downloads()

# Kick off
if __name__ == "__main__":
    main()
    with open("bad_urls.txt", "w") as f:
        for url in BAD_URLS:
            f.write("%s\n" % url)
