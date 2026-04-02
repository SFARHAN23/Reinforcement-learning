from CarModel_Kinematic import CarModelClass
import math

print("--- Testing Steering Death at 90 m/s ---")
# Start at 90 m/s
car = CarModelClass([0, 0, 0], 85.0) # start just below SPD_MAX

# Apply full throttle to let it stabilize 
for i in range(10):
    car.step([1.0, 0.0]) # full throttle, no steer

print(f"Speed: {car.spd:.2f} m/s, max limit: ~{car.acc_sum:.2f} vs acc_limit. Let's apply 5% steer.")

# Now apply small steer
fail_step = -1
for i in range(100):
    car.step([1.0, 0.05]) # only 5% steer rate
    if car.check_acc():
        fail_step = i
        break

print(f"Failed at step {fail_step} with lat_acc={car.lat_acc:.2f} m/s^2, long_acc={car.long_acc:.2f} m/s^2, steer_angle={car.steer:.4f} rad ({math.degrees(car.steer):.2f} deg)")
