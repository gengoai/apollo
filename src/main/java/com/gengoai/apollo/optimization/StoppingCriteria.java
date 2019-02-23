package com.gengoai.apollo.optimization;

import com.gengoai.Stopwatch;
import com.gengoai.logging.Logger;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.function.IntToDoubleFunction;

/**
 * The type Termination criteria.
 *
 * @author David B. Bracewell
 */
public final class StoppingCriteria implements Serializable {
   private final LinkedList<Double> history = new LinkedList<>();
   private int maxIterations = 100;
   private int historySize = 5;
   private double tolerance = 1e-6;
   private final String criteriaName;
   private int reportInterval = -1;
   private Logger logger = Logger.getGlobalLogger();

   private StoppingCriteria(String criteriaName) {
      this.criteriaName = criteriaName;
   }

   /**
    * Create termination criteria.
    *
    * @return the termination criteria
    */
   public static StoppingCriteria create() {
      return new StoppingCriteria("loss");
   }

   public static StoppingCriteria create(String criteriaName) {
      return new StoppingCriteria(criteriaName);
   }

   /**
    * Check boolean.
    *
    * @param sumLoss the sum loss
    * @return the boolean
    */
   public boolean check(double sumLoss) {
      boolean converged = false;
      if (!Double.isFinite(sumLoss)) {
         System.err.println("Non Finite loss, aborting");
         return true;
      }
      if (history.size() >= historySize) {
         converged = Math.abs(sumLoss - history.removeLast()) <= tolerance;
         Iterator<Double> itr = history.iterator();
         while (converged && itr.hasNext()) {
            double n = itr.next();
            converged = Math.abs(sumLoss - n) <= tolerance //loss in tolerance
                           || sumLoss > n; //or we got worse
         }
      }
      history.addFirst(sumLoss);
      return converged;
   }

   /**
    * History size int.
    *
    * @return the int
    */
   public int historySize() {
      return this.historySize;
   }

   /**
    * History size termination criteria.
    *
    * @param historySize the history size
    * @return the termination criteria
    */
   public StoppingCriteria historySize(int historySize) {
      this.historySize = historySize;
      return this;
   }

   /**
    * Last loss double.
    *
    * @return the double
    */
   public double lastLoss() {
      return history.getFirst();
   }


   /**
    * Max iterations int.
    *
    * @return the int
    */
   public int maxIterations() {
      return this.maxIterations;
   }

   /**
    * Max iterations termination criteria.
    *
    * @param maxIterations the max iterations
    * @return the termination criteria
    */
   public StoppingCriteria maxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
   }

   /**
    * Tolerance double.
    *
    * @return the double
    */
   public double tolerance() {
      return this.tolerance;
   }

   /**
    * Tolerance termination criteria.
    *
    * @param tolerance the tolerance
    * @return the termination criteria
    */
   public StoppingCriteria tolerance(double tolerance) {
      this.tolerance = tolerance;
      return this;
   }


   public String criteriaName() {
      return this.criteriaName;
   }


   public int reportInterval() {
      return reportInterval;
   }

   public StoppingCriteria reportInterval(int reportInterval) {
      this.reportInterval = reportInterval;
      return this;
   }


   public Logger logger() {
      return logger;
   }

   public StoppingCriteria logger(Logger logger) {
      this.logger = logger == null ? Logger.getGlobalLogger() : logger;
      return this;
   }


   public int untilTermination(IntToDoubleFunction iteration) {
      Stopwatch sw = Stopwatch.createStopped();
      double loss = 0;
      for (int i = 0; i < maxIterations; i++) {
         sw.reset();
         sw.start();
         loss = iteration.applyAsDouble(i);
         sw.stop();
         if (check(loss)) {
            logger.info("iteration {0}: {1}={2}, time={3}, Converged", (i + 1), criteriaName, loss, sw);
            return i;
         }
         if (reportInterval > 0 && (i + 1) % reportInterval == 0) {
            logger.info("iteration {0}: {1}={2}, time={3}", (i + 1), criteriaName, loss, sw);
         }
      }
      if (reportInterval > 0 && (maxIterations + 1) % reportInterval != 0) {
         logger.info("iteration {0}: {1}={2}, time={3}, Max. Iterations Reached", maxIterations, criteriaName, loss,
                     sw);
      }
      return maxIterations;
   }

}// END OF TerminationCriteria
