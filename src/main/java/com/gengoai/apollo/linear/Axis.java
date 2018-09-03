package com.gengoai.apollo.linear;

/**
 * Enumeration of the possible axises in an NDArray
 *
 * @author David B. Bracewell
 */
public enum Axis {
   /**
    * Row axis.
    */
   ROW(0),
   /**
    * Column axis.
    */
   COLUMN(1),
   /**
    * Kernel axis.
    */
   KERNEL(2),
   /**
    * Channel axis.
    */
   CHANNEL(3);

   /**
    * The index of the axis in a shape array.
    */
   public final int index;

   Axis(int index) {
      this.index = index;
   }


   /**
    * Sets the value at this axis's index in the given indices array to the given value. Typical usage is to change the
    * dimension for a given axis, e.g. <code>Axis.ROW.set(ndarray.shape(), 1)</code> will set the row index in the
    * returned shape array to have dimension <code>1</code>.
    *
    * @param indices the indices
    * @param value   the value
    * @return the modified indices array
    */
   public int[] set(int[] indices, int value) {
      indices[index] = value;
      return indices;
   }

   /**
    * Selects the value corresponding to this axis in the given set of indices.
    *
    * @param indices the indices
    * @return the value in the indices array corresponding to this axis
    */
   public int select(int... indices) {
      switch (this) {
         case ROW:
            return indices.length > 0 ? indices[0] : 1;
         case COLUMN:
            return indices.length > 1 ? indices[1] : 1;
         case KERNEL:
            return indices.length > 2 ? indices[2] : 1;
         default:
            return indices.length > 3 ? indices[3] : 1;
      }
   }

   /**
    * Gets the transposed axis corresponding to this one (e.g. COLUMN for ROW)
    *
    * @return the transposed axis
    */
   public Axis T() {
      switch (this) {
         case ROW:
            return COLUMN;
         case COLUMN:
            return ROW;
         case KERNEL:
            return CHANNEL;
         default:
            return KERNEL;
      }
   }

   /**
    * Checks if the axis is a ROW or COLUMN
    *
    * @return True if is a ROW or COLUMN
    */
   public boolean isRowOrColumn() {
      return this == ROW || this == COLUMN;
   }

   /**
    * Checks if the axis is a KERNEL or CHANNEL
    *
    * @return True if is a KERNEL or CHANNEL
    */
   public boolean isKernelOrChannel() {
      return this == KERNEL || this == CHANNEL;
   }

   /**
    * Gets the axis enum value for the given axis index
    *
    * @param index the axis index
    * @return the axis
    */
   public static Axis valueOf(int index) {
      switch (index) {
         case 0:
            return ROW;
         case 1:
            return COLUMN;
         case 2:
            return KERNEL;
         case 3:
            return CHANNEL;
      }
      throw new IllegalArgumentException("Axis (" + index + ") is undefined.");
   }

}//END OF Axis
