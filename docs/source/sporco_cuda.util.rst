sporco_cuda.util module
=======================


.. py:function:: device_count()

   Get the number of CUDA GPU devices installed on the host system.

   Returns
   -------
   ndev : int
      Number of installed deviced


.. py:function:: current_device(id=None)

   Get or set the current CUDA GPU device. The current device is not set
   if `id` is None

   Parameters
   ----------
   id : int or None, optional (default None)
     Device number of device to be set as current device

   Returns
   -------
   id : int
     Device number of current device


.. py:function:: memory_info()

   Get memory information for the current CUDA GPU device.

   Returns
   -------
   free : int
     Free memory in bytes
   total : int
      Total memory in bytes
