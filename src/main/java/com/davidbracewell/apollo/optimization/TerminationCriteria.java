package com.davidbracewell.apollo.optimization;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public final class TerminationCriteria implements Serializable {
   private final LinkedList<Double> history = new LinkedList<>();
   @Getter
   @Setter
   private int maxIterations = 100;
   @Getter
   @Setter
   private int historySize = 5;
   @Getter
   @Setter
   private double tolerance = 1e-6;

   public static TerminationCriteria create() {
      return new TerminationCriteria();
   }

   public boolean check(double sumLoss) {
      boolean converged = false;
      if (history.size() >= historySize) {
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
