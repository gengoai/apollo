package com.gengoai.apollo.linear;

import com.gengoai.tuple.Tuple3;

import java.util.Arrays;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.tuple.Tuples.$;

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

   public boolean is(Axis... choices) {
      return Arrays.stream(choices).anyMatch(c -> c == this);
   }


   public Tuple3<Axis,Integer,Integer> range(int start, int end){
      checkArgument(end >= start, "End must be > start");
      checkArgument(start >= 0, "Start must be non-negative");
      return $(this,start,end);
   }

   public int[] set(int[] indices, int value) {
      indices[ordinal] = value;
      return indices;
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
