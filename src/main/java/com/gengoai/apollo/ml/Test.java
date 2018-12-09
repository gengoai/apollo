package com.gengoai.apollo.ml;

import java.util.function.Supplier;

/**
 * @author David B. Bracewell
 */
public class Test {
   static class P {

      public void f(FitParameters consumer) {

      }

      public FitParameters getParameters() {
         return null;
      }

   }

   static class C extends FitParameters {
      public int i = 100;
   }

   static class Q extends P {

      @Override
      public void f(FitParameters consumer) {
      }

      public <T extends FitParameters> void f(Supplier<T> supplier) {

      }


      @Override
      public C getParameters() {
         return new C();
      }

   }


   public static void main(String[] args) throws Exception {
      Q q = new Q();
      q.f(() -> {
         C c = q.getParameters();
         c.numFeatures = 1;
         c.i = 2;
         return c;
      });
   }


}//END OF Test
