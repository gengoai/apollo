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

package com.davidbracewell.apollo;

import com.davidbracewell.function.SerializableDoubleUnaryOperator;

/**
 * <p>A unary operator that also has methodology to calculate the derivative of the function.</p>
 *
 * @author David B. Bracewell
 */
@FunctionalInterface
public interface DifferentiableFunction extends SerializableDoubleUnaryOperator {
   /**
    * Constant used in approximating the derivative
    */
   double H = 1E-5;
   /**
    * Constant used in approximating the derivative
    */
   double TWO_H = 2.0 * H;

   /**
    * Calculates the derivative of the function at the given value
    *
    * @param value the value to calculate the derivative at
    * @return the derivative
    */
   default double derivative(double value) {
      return (applyAsDouble(value + H) - applyAsDouble(value - H)) / TWO_H;
   }

}//END OF DifferentiableFunction
