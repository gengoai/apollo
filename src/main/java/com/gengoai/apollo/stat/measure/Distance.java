package com.gengoai.apollo.stat.measure;


import com.gengoai.math.Math2;
import com.gengoai.apollo.linear.NDArray;
import lombok.NonNull;

/**
 * <p>Commonly used distance measures.</p>
 *
 * @author dbracewell
 */
public enum Distance implements DistanceMeasure {
   /**
    * <a href="https://en.wikipedia.org/wiki/Euclidean_distance">Euclidean distance</a>
    */
   Euclidean {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return Math.sqrt((v1.dot(v1) + v2.dot(v2)) - (2.0 * v1.dot(v2)));
      }
   },
   /**
    * Variation on Euclidean distance that doesn't take the square root of the sum of squared differences
    */
   SquaredEuclidean {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return (v1.dot(v1) + v2.dot(v2)) - (2.0 * v1.dot(v2));
      }

   },
   /**
    * <a href="https://en.wiktionary.org/wiki/Manhattan_distance">Manhattan distance</a>
    */
   Manhattan {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return v1.sub(v2).norm1();
      }


   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Hamming_distance">Hamming Distance</a>
    */
   Hamming {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return v1.map(v2, (d1, d2) -> d1 != d2 ? 1 : 0).sum();
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Earth_mover%27s_distance">Earth mover's distance</a>
    */
   EarthMovers {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         double last = 0;
         double sum = 0;
         for (int i = 0; i < v1.length(); i++) {
            double d1 = v1.get(i);
            double d2 = v2.get(i);
            double dist = (d1 + last) - d2;
            sum += Math.abs(dist);
            last = dist;
         }
         return sum;
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Chebyshev_distance">Chebyshev distance</a>
    */
   Chebyshev {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return v1.map(v2, (d1, d2) -> Math.abs(d1 - d2)).max();
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback–Leibler divergence</a>
    */
   KLDivergence {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return v1.map(v2, (d1, d2) -> d1 * Math2.safeLog(d1 / d2)).sum();
      }
   },
   /**
    * <a href="https://en.wikipedia.org/wiki/Angular_distance">Angular distance</a>
    */
   Angular {
      @Override
      public double calculate(@NonNull NDArray v1, @NonNull NDArray v2) {
         return Math.acos(Similarity.Cosine.calculate(v1, v2)) / Math.PI;
      }
   }


}//END OF Distance
