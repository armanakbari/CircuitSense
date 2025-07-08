.title Active DC Circuit
R1 0 1 2
L1 0 3 96
L2 1 2 70
* Non-inverting amplifier with gain 50
E1 1 0 4 0 50
* Integrator (high gain approximation)
E1_1 4 0 0 3 -1000
* Summing amplifier with gain -17
E1_2 3 0 0 5 -17
R2 2 4 69
VI1 4 5 0
* Non-inverting amplifier with gain 33
E2 6 0 2 0 33
V1 5 6 DC 59 AC 59


.control
ac dec 10 1 100k
print vm(1) vp(1) ; AC magnitude and phase of U8
print im(VI1) ip(VI1) ; AC magnitude and phase of I5
.endc
.end
