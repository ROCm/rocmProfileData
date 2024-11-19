.headers on
.print '### Top-10 kernels'
select * from top limit 10;

.print '### Contribution of matrix multiplications'
select sum(Percentage) from top where Name like 'Cijk%';
.print '### Contribution of elementwise kernels'
select sum(Percentage) from top where Name like '%elementwise%';
.print '### Contribution of collective communication'
select sum(Percentage) from top where Name like '%ccl%';
.print '### Contribution of reduction kernels'
select sum(Percentage) from top where Name like '%reduce_kernel%';

.print '### Busy time on GPU'
select * from busy;
