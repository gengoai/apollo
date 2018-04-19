/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.gengoai.apollo.stat.distribution;

import java.io.Serializable;

/**
 * <p>A density function involving two random variables <code>X</code> and <code>Y</code>.</p>
 *
 * @author David B. Bracewell
 */
public interface BivariateDistribution extends Serializable {

   /**
    * Probability of event with <code>X=x</code> and <code>Y=y</code>
    *
    * @param x the x variable
    * @param y the y variable
    * @return the probability of event with <code>X=x</code> and <code>Y=y</code>
    */
   double probability(int x, int y);

   /**
    * Log probability of event with <code>X=x</code> and <code>Y=y</code>
    *
    * @param x the x variable
    * @param y the y variable
    * @return the log probability of event with <code>X=x</code> and <code>Y=y</code>
    */
   default double logProbability(int x, int y) {
      return Math.log(probability(x, y));
   }


}//END OF BivariateDistribution
