# pydrs
very experimental python wrapper of DRS4 software. At the moment it allows reading from one channel on a DRS4 Evaluation Board V5, while removing spikes.

On initialization config file "drsosc.cfg" created by drsosc is read and (some of the) settings are applied accordingly.

To build, DRS.cpp, DRS.h, mxml.c, strlcpy.c, averager.cpp, musbstd.c need to be present and numpy is needed. Then it can be build using the setup.py script.
