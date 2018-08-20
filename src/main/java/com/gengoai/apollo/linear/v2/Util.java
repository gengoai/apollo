package com.gengoai.apollo.linear.v2;

/**
 * @author David B. Bracewell
 */
final class Util {
   private Util() {
      throw new IllegalAccessError();
   }

   static int[] ensureCorrectIndicies(int... dimensions) {
      int[] shape = new int[]{1, 1, 1, 1};
      System.arraycopy(dimensions, 0, shape, 0, dimensions.length);
      return shape;
   }

   static long index(int ax1, int dimAx1, int ax2, int dimAx2,
                     int ax3, int dimAx3, int ax4, int dimAx4
                    ) {
      int sliceIndex = index(ax3, dimAx3, ax4, dimAx4);
      int matrixIndex = index(ax1, dimAx1, ax2, dimAx2);
      int matrixLength = dimAx1 * dimAx2;
      int sliceLength = dimAx3 * dimAx4;
      return index(matrixIndex, matrixLength, sliceIndex, sliceLength);
   }

   static int[] reverseIndex(long index, int dimAx1, int dimAx2,
                             int dimAx3, int dimAx4
                            ) {
      int matrixLength = dimAx1 * dimAx2;
      int sliceLength = dimAx3 * dimAx4;
      int[] imd = reverseIndex(index, matrixLength, sliceLength);
      int[] matrix = reverseIndex(imd[0], dimAx1, dimAx2);
      int[] slice = reverseIndex(imd[1], dimAx3, dimAx4);
      return new int[]{
         matrix[0], matrix[1],
         slice[0], slice[1]
      };
   }


   static int[] reverseIndex(long index, int dimAx1, int dimAx2) {
      return new int[]{
         (int) index % dimAx1,
         (int) index / dimAx1
      };
   }

   static int index(int ax1, int dimAx1, int ax2, int dimAx2) {
      return ax1 + (dimAx1 * ax2);
   }

   static int sliceIndex(int kernel, int channel, int numChannel) {
      return kernel + (numChannel * channel);
   }

   static int[] reverseSliceIndex(int index, int numKernel) {
      return new int[]{index % numKernel, index / numKernel};
   }

   static int rowColumn(int kernel, int channel, int numChannel) {
      return kernel + (numChannel * channel);
   }

   static int[] reverseRowColumnIndex(int index, int numKernel) {
      return new int[]{index % numKernel, index / numKernel};
   }


}//END OF Util
