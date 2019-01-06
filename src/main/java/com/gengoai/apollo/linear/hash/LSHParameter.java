package com.gengoai.apollo.linear.hash;

import com.gengoai.Copyable;
import com.gengoai.Parameters;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class LSHParameter implements Parameters<LSHParameter>, Serializable, Copyable<LSHParameter> {
   private static final long serialVersionUID = 1L;
   public int bands = 5;
   public int buckets = 20;
   public String signature = "COSINE";
   public int dimension = 1;
   public int signatureSize = -1;
   public int max_w = 100;
   public double threshold = 0.5;

   @Override
   public LSHParameter copy() {
      return Copyable.deepCopy(this);
   }
}//END OF LSHP
