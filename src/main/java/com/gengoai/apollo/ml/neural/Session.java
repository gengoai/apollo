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
 *
 */

package com.gengoai.apollo.ml.neural;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.conversion.Cast;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class Session {

   public NDArray run(Node operation, Map<Node, NDArray> feedDict) {
      List<Node> postOrder = new ArrayList<>();
      postOrderTraversal(operation, postOrder);

      for (Node node : postOrder) {

         if (node instanceof Placeholder) {

            Placeholder placeholder = Cast.as(node);
            placeholder.output = feedDict.get(node);

         } else if (node instanceof Variable) {

            Variable variable = Cast.as(node);
            variable.output = node.value;

         } else {

            Operation op = Cast.as(node);
            node.output = op.compute();


         }

      }


      return operation.output;
   }


   private void postOrderTraversal(Node node, List<Node> postOrder) {
      if (node instanceof Operation) {
         Operation op = Cast.as(node);
         for (Node inputNode : op.inputNodes) {
            postOrderTraversal(inputNode, postOrder);
         }
      }
      postOrder.add(node);
   }


}//END OF Session
