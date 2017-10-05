package com.davidbracewell.apollo;

/**
 * Methods for comparing numeric (double) values.
 *
 * @author David B. Bracewell
 */
public enum NumericComparison {
   /**
    * Is <code>beingCompared</code> greater than <code>comparedAgainst</code>
    */
   GT {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared > comparedAgainst;
      }
   },
   /**
    * Is <code>beingCompared</code> greater than or equal to <code>comparedAgainst</code>
    */
   GTE {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared >= comparedAgainst;
      }
   },
   /**
    * Is <code>beingCompared</code> less than <code>comparedAgainst</code>
    */
   LT {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared < comparedAgainst;
      }
   },
   /**
    * Is <code>beingCompared</code> less than or equal to <code>comparedAgainst</code>
    */
   LTE {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared <= comparedAgainst;
      }
   },
   /**
    * Is <code>beingCompared</code> equal to <code>comparedAgainst</code>
    */
   EQ {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared == comparedAgainst;
      }
   },
   /**
    * Is <code>beingCompared</code> not equal to <code>comparedAgainst</code>
    */
   NE {
      @Override
      public boolean compare(double beingCompared, double comparedAgainst) {
         return beingCompared != comparedAgainst;
      }
   };

   /**
    * Compares two given numeric values
    *
    * @param beingCompared   The number being compared
    * @param comparedAgainst The number being compared against
    * @return true if the inequality holds
    */
   public abstract boolean compare(double beingCompared, double comparedAgainst);

}// END OF Inequality
