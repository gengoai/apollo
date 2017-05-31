package com.davidbracewell.apollo.optimization;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * @author David B. Bracewell
 */
public final class TerminationCriteria implements Serializable {

   private final LinkedList<Double> history = new LinkedList<>();
   private int minIterations = 5;
   private double tolerance = 1e-6;

   public boolean check(double sumLoss) {
      boolean converged = false;
      if (history.size() >= minIterations) {
         converged = Math.abs(sumLoss - history.removeLast()) <= tolerance;
         Iterator<Double> itr = history.iterator();
         while (converged && itr.hasNext()) {
            converged = Math.abs(sumLoss - itr.next()) <= tolerance;
         }
      }
      history.addFirst(sumLoss);
      return converged;
   }

   public double lastLoss() {
      return history.getFirst();
   }


}// END OF TerminationCriteria
