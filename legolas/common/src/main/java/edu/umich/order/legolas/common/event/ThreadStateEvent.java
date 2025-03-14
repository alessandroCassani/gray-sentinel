/*
 *  @author Haoze Wu <haoze@jhu.edu>
 *
 *  The Legolas Project
 *
 *  Copyright (c) 2024, University of Michigan, EECS, OrderLab.
 *      All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.umich.order.legolas.common.event;

import edu.umich.order.legolas.common.asm.AbstractState;
import edu.umich.order.legolas.common.record.RecordWriter;
import java.io.IOException;

/**
 * TODO: more general definition of state
 */
public class ThreadStateEvent extends NodeEvent {
    public String threadName;
    public int instanceId;
    public String stateMachineName;
    public AbstractState state;

    public ThreadStateEvent(long nano, int serverId, String threadName,
            int instanceId, String stateMachineName, AbstractState state) {
        super(nano, serverId);
        this.threadName = threadName;
        this.instanceId = instanceId;
        this.stateMachineName = stateMachineName;
        this.state = state;
    }

    @Override
    public void dump(final RecordWriter writer) throws IOException {
        super.dump(writer);
        writer.append(writer.getStateMachine(stateMachineName));
        writer.append(writer.getOp(state.methodSig));
        writer.append(state.id);
        writer.append(instanceId);
    }

    @Override
    public int getType() {
        return 3;
    }
}
