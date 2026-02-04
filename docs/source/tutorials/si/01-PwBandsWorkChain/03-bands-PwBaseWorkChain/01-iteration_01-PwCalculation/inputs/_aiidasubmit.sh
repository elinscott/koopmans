#!/bin/bash
exec > _scheduler-stdout.txt
exec 2> _scheduler-stderr.txt


'mpirun' '-np' '1' '/usr/local/bin/pw.x' '-in' 'aiida.in'  > 'aiida.out'
