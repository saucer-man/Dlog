#!/usr/bin/env python

import sys
import pyparsing

from pyparsing import Word
from pyparsing import alphas
from pyparsing import Suppress
from pyparsing import Combine
from pyparsing import nums
from pyparsing import string
from pyparsing import Optional
from pyparsing import Regex
#from pyparsing import Literal
#from pyparsing import delimitedList

from time import strftime


class syslogparser(object):
    def __init__(self):
        # timestamp
        month = Word(string.ascii_uppercase, string.ascii_lowercase, exact=3)
        day   = Word(nums)
        hour  = Combine(Word(nums) + ":" + Word(nums) + ":" + Word(nums))
        timestamp = Combine(month + " " + day + " " + hour)

        # hostname
        hostname = Word(alphas + nums + "_" + "-" + ".")

        # appname
        appword = Word(alphas + nums + "/" + "-" + "_" + "." + "(" + ")" + "[" + "]")
        appname = Combine(appword + Optional(" (" + appword))

        # ProcessID
        #pid = Word(Suppress("[") + Word(nums) + Suppress("]"))

        # message
        message = Combine(Suppress(":") + Regex(".*"))
      
        self._pattern = timestamp + hostname + appname + message

    def parse(self, line):

        parsed = self._pattern.parseString(line)

        payload              = {}
        #payload["timestamp"] = strftime("%Y-%m-%d %H:%M:%S")
        payload["timestamp"] = parsed[0]
        payload["hostname"]  = parsed[1]
        payload["appname"]   = parsed[2]
        payload["message"]   = parsed[3]
        #payload["pid"]       = parsed[4]

        return payload


def main():
    parser = syslogparser()

    for i in sys.stdin.readlines():
        sub = i.strip('\n')
        fields = parser.parse(sub)
        print(fields)
  
if __name__ == "__main__":
    main()


