# === program the chip ======================================================

###################
# Import libraries
###################
import AERmn256r1
import time

print "Program connections weights ..."
AERmn256r1.load_weight_matrix_programmable(conf_mn256r1.matrix_programmable_w)
time.sleep(1)
print "Program connections exc/inh ..."
AERmn256r1.load_matrix_exc_inh(conf_mn256r1.matrix_programmable_exc_inh)
time.sleep(1)
print "Program recurrent plastic connections ..."
AERmn256r1.load_connections_matrix_plastic(conf_mn256r1.matrix_learning_rec)
time.sleep(1)
print "Program connections ..."
AERmn256r1.load_connections_matrix_programmable(conf_mn256r1.matrix_programmable_rec)
time.sleep(1)
print "Program plastic weights ..."
AERmn256r1.load_weight_matrix_plastic(conf_mn256r1.matrix_learning_pot)
time.sleep(1)


