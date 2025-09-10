from policy.policy_runner import ResidualPolicyRunner
from common.circular_buffer import CircularBuffer, IsaacCircularBuffer
import torch
import numpy as np


def test_buffer_order():
    """Test that both buffers return data in the same order"""
    print("="*60)
    print("CIRCULAR BUFFER ORDER COMPARISON TEST")
    print("="*60)
    
    # Test parameters
    max_len = 5
    data_dim = 3
    batch_size = 1
    device = "cpu"
    
    # Initialize buffers
    torch_buffer = IsaacCircularBuffer(max_len, batch_size, device)
    numpy_buffer = CircularBuffer(max_len, (data_dim,))
    
    print(f"Buffer size: {max_len}, Data dim: {data_dim}")
    print()
    
    # Test case 1: Add data step by step
    print("TEST 1: Adding data step by step")
    print("-" * 40)
    
    test_data = []
    for step in range(7):  # Add more than buffer size to test wraparound
        # Create test data
        data_value = step + 1
        torch_data = torch.tensor([[data_value, data_value * 10, data_value * 100]], dtype=torch.float32, device=device)
        numpy_data = np.array([data_value, data_value * 10, data_value * 100], dtype=np.float32)
        test_data.append(numpy_data.copy())
        
        # Add to buffers
        torch_buffer.append(torch_data)
        numpy_buffer.append(numpy_data)
        
        print(f"Step {step}: Added {numpy_data}")
        
        # Compare outputs when we have enough data
        if step >= 2:  # Start comparing from step 2
            torch_length = torch_buffer.current_length[0].item()
            numpy_length = numpy_buffer.current_length
            
            print(f"  Torch buffer length: {torch_length}")
            print(f"  Numpy buffer length: {numpy_length}")
            
            if torch_length >= 3 and numpy_length >= 3:
                # Get last 3 steps from torch buffer
                torch_full_buffer = torch_buffer.buffer[0]  # Remove batch dimension
                torch_last_3 = torch_full_buffer[-3:].cpu().numpy()
                
                # Get last 3 steps from numpy buffer
                numpy_last_3 = numpy_buffer.get_history(3)
                
                print(f"  Torch last 3 steps:")
                for i, data in enumerate(torch_last_3):
                    print(f"    [{i}] {data}")
                
                print(f"  Numpy last 3 steps:")
                for i, data in enumerate(numpy_last_3):
                    print(f"    [{i}] {data}")
                
                # Check if they match
                if np.allclose(torch_last_3, numpy_last_3, atol=1e-6):
                    print(f"  ✅ ORDER MATCHES")
                else:
                    print(f"  ❌ ORDER MISMATCH!")
                    print(f"     Difference: {np.abs(torch_last_3 - numpy_last_3)}")
        print()
    
    print("\n" + "="*60)
    print("TEST 2: Buffer wraparound behavior")
    print("-" * 40)
    
    # Reset buffers
    torch_buffer.reset(batch_ids=None)
    numpy_buffer.reset()
    
    # Fill buffer completely and then add more
    for step in range(max_len + 3):  # Exceed buffer size
        data_value = step + 10  # Different values to distinguish
        torch_data = torch.tensor([[data_value, data_value * 2, data_value * 3]], dtype=torch.float32, device=device)
        numpy_data = np.array([data_value, data_value * 2, data_value * 3], dtype=np.float32)
        
        torch_buffer.append(torch_data)
        numpy_buffer.append(numpy_data)
        
        if step >= max_len:  # After buffer is full
            print(f"Step {step}: Added {numpy_data} (buffer full, wrapping)")
            
            # Compare full buffers
            torch_full = torch_buffer.buffer[0].cpu().numpy()
            numpy_full = numpy_buffer.get_history(max_len)
            
            print(f"  Torch buffer (oldest to newest):")
            for i, data in enumerate(torch_full):
                print(f"    [{i}] {data}")
            
            print(f"  Numpy buffer (oldest to newest):")
            for i, data in enumerate(numpy_full):
                print(f"    [{i}] {data}")
            
            if np.allclose(torch_full, numpy_full, atol=1e-6):
                print(f"  ✅ WRAPAROUND ORDER MATCHES")
            else:
                print(f"  ❌ WRAPAROUND ORDER MISMATCH!")
            print()
    
    print("\n" + "="*60)
    print("TEST 3: Individual access pattern")
    print("-" * 40)
    
    # Test individual access using __getitem__ for torch buffer
    for steps_back in range(min(3, numpy_buffer.current_length)):
        # Torch buffer access (need to create tensor key)
        key = torch.tensor([steps_back], device=device)
        torch_item = torch_buffer[key][0].cpu().numpy()  # Remove batch dimension
        
        # Numpy buffer access
        numpy_item = numpy_buffer.get_recent(steps_back)
        
        print(f"  {steps_back} steps back:")
        print(f"    Torch: {torch_item}")
        print(f"    Numpy: {numpy_item}")
        
        if np.allclose(torch_item, numpy_item, atol=1e-6):
            print(f"    ✅ MATCH")
        else:
            print(f"    ❌ MISMATCH!")
        print()
    
    print("="*60)
    print("TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    test_buffer_order()
