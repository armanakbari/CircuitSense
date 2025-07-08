.title Active DC Circuit
* Integrator (high gain approximation)
E1 1 0 0 0 -1000
* Non-inverting amplifier with gain 28
E1_1 0 0 2 0 28
* Unity gain buffer
E1_2 1 0 3 0 1
R4 2 3 28
L1 2 4 52
R1 3 5 15
R5 5 4 62
R2 4 6 32
R3 5 7 28
V1 6 7 DC 1 AC 1


.control
ac dec 10 1 100k
print vm(1) vp(1) ; AC magnitude and phase of U4
.endc
.end
