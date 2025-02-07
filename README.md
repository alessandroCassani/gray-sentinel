Gray failure is a term used primarily in the context of network systems and computing to
describe a scenario where a component of the system is partially failing, yet still operational
to some extent. This partial failure can cause unpredictable behavior because the system
does not completely fail, nor does it operate normally, making detection and diagnosis
challenging. The "gray" aspect refers to its ambiguity and its position between full
operational status and complete failure.
Gray failures are notoriously difficult to detect in datacenters because they subtly
degrade system performance without causing complete system outages. These failures
present a significant operational challenge as they can lead to unforeseen performance
bottlenecks and reliability issues. Traditional monitoring tools are often not equipped to
detect these failures until they manifest more severe symptoms, at which point the
damage might already be impacting services.
This project proposes the development of the Predictive Gray Failure Detection System
(PGFDS), a sophisticated monitoring solution designed to identify and predict gray
failures by analyzing side-channel metrics such as memory requests, I/O requests,
system call frequencies, instruction per cycle (IPC), and power consumption. The
innovative aspect of this system is its use of metrics that are typically considered indirect
indicators of system health, similar to the techniques used in side-channel attacks,
which derive insights from seemingly unrelated data.
