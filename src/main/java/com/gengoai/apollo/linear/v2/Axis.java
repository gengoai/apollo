package com.gengoai.apollo.linear.v2;

/**
 * @author David B. Bracewell
 */
public enum Axis {
   ROW(0),
   COLUMN(1),
   KERNEL(2),
   CHANNEL(3);

   public final int ordinal;

   Axis(int ordinal) {
      this.ordinal = ordinal;
   }

   public int select(int... indices) {
      switch (this) {
         case ROW:
            return indices[0];
         case COLUMN:
            return indices[1];
         case KERNEL:
            return indices.length == 2 ? indices[0] : indices[2];
         default:
            return indices.length == 2 ? indices[1] : indices[3];
      }
   }

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

   public boolean isRowOrColumn() {
      return this == ROW || this == COLUMN;
   }

   public boolean isKernelOrChannel() {
      return this == KERNEL || this == CHANNEL;
   }

   public static Axis valueOf(int ordinal) {
      switch (ordinal) {
         case 0:
            return ROW;
         case 1:
            return COLUMN;
         case 2:
            return KERNEL;
         case 3:
            return CHANNEL;
      }
      throw new IllegalArgumentException("Axis (" + ordinal + ") is undefined.");
   }

}//END OF Axis
