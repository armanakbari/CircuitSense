.title Active DC Circuit
* Inverting amplifier with gain -5
E1 0 0 0 1 -5
* Non-inverting amplifier with gain 22
E1_1 0 0 2 0 22
* Differentiator (high gain approximation)
E1_2 3 0 0 1 -1000
C1 2 4 15
R1 5 3 71
R3 5 4 7
R2 5 6 8
V1 4 6 DC 34 AC 34


.control
ac dec 10 1 100k
.endc
.end
