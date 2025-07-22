#!/usr/bin/env python3

import multiprocessing
import time

def run_with_timeout(func, timeout_seconds):
    """Test version of multiprocessing timeout"""
    def target_func(queue, func):
        try:
            result = func()
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', str(e)))
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_func, args=(queue, func))
    process.start()
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        print(f"⏰ Process timed out after {timeout_seconds}s, terminating...")
        process.terminate()
        process.join()
        return None, f"Timeout after {timeout_seconds}s"
    
    if queue.empty():
        return None, "Process ended without result"
    
    result_type, result = queue.get()
    if result_type == 'success':
        return result, None
    else:
        return None, result

def slow_computation():
    """Simulate a slow symbolic computation"""
    print("Starting slow computation...")
    time.sleep(10)  # Simulate 10 seconds of work
    return "Computation completed"

def fast_computation():
    """Simulate a fast computation"""
    print("Starting fast computation...")
    time.sleep(1)
    return "Fast computation completed"

def test_timeouts():
    # Set multiprocessing method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print("🧪 Testing multiprocessing timeout mechanism...")
    
    # Test 1: Fast computation should succeed
    print("\n1️⃣ Testing fast computation (should succeed):")
    result, error = run_with_timeout(fast_computation, 5)
    if error:
        print(f"❌ Unexpected error: {error}")
    else:
        print(f"✅ Success: {result}")
    
    # Test 2: Slow computation should timeout
    print("\n2️⃣ Testing slow computation (should timeout):")
    result, error = run_with_timeout(slow_computation, 3)
    if error:
        print(f"✅ Expected timeout: {error}")
    else:
        print(f"❌ Unexpected success: {result}")
    
    print("\n🎯 Timeout mechanism test completed!")

if __name__ == "__main__":
    test_timeouts() 